import numpy as np  
import torch  
import os  
from utils.utils import *  
# from utils.core_utils import Accuracy_Logger, EarlyStopping  
import pandas as pd
from sklearn.preprocessing import label_binarize  
from sklearn.metrics import roc_auc_score, roc_curve, f1_score  
from sklearn.metrics import auc as calc_auc  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count
  
# 三个函数代码保持不变  
def train_loop_clam_lt(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,
                       bag_stats_weight=0.1, contrastive_weight=0.05, accumulation_steps=8,
                       attention_weight=0.01, use_entropy=True, use_smoothing=True):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # 用于对比学习的累积  
    accumulated_embeddings = []  
    accumulated_labels = [] 

    print('\n')
    for batch_idx, (data, label, coords) in enumerate(loader):  # 注意这里需要coords  
        data, label, coords = data.to(device), label.to(device), coords.to(device)  
        logits, Y_prob, Y_hat, _, instance_dict = model(data, coords=coords, label=label, instance_eval=True)
        A = F.softmax(_, dim=1)  # 归一化的 attention weights

         # 获取 density embedding  
        bag_stats_true = instance_dict['bag_stats_true']  
        density_embedding = model.density_projector(bag_stats_true)  # (128,)

        acc_logger.log(Y_hat, label)
        # 主损失
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']

        # -----[1]-----Density regression loss  
        bag_stats_true = instance_dict['bag_stats_true']  
        bag_stats_pred = instance_dict['bag_stats_pred']  
        # density_loss = F.mse_loss(bag_stats_pred, bag_stats_true) 这里应该叫做bag_stats_loss
        bag_stats_loss = F.mse_loss(bag_stats_pred, bag_stats_true)

        # -----[2]-----Attention regularization  
        attention_loss = 0.0  
        if use_entropy:  
            # 只对正样本鼓励一定的 spread  
            if label.item() == 1:  
                attention_loss += compute_attention_entropy_loss(A, label, encourage_spread=True)  
          
        if use_smoothing:  
            attention_loss += compute_attention_smoothing_loss(A, coords, sigma=100.0)  

        # -----[3]-----获取 bag-level embedding（用于对比学习）  
        # 关键：使用 .detach() 分离计算图  
        if 'bag_embedding' in instance_dict:  
            bag_embedding = instance_dict['bag_embedding'].detach()  
            accumulated_embeddings.append(bag_embedding)  
            accumulated_labels.append(label.item())  

         # 每 accumulation_steps 计算一次对比损失
        contrastive_loss = 0.0 
        if (batch_idx + 1) % accumulation_steps == 0 and len(accumulated_embeddings) > 1:  
            # 将累积的 embeddings 转换为 tensor  
            embeddings_tensor = torch.stack(accumulated_embeddings)  # (B, embed_dim)  
            labels_tensor = torch.tensor(accumulated_labels, device=device)  
              
            # 计算对比学习损失  
            contrastive_loss = compute_contrastive_loss(embeddings_tensor, labels_tensor)  
              
            # 清空累积  
            accumulated_embeddings = []  
            accumulated_labels = []  

        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        # 总损失
        # total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
        total_loss = (bag_weight * loss + 
                      (1-bag_weight) * instance_loss + 
                      bag_stats_weight * bag_stats_loss + 
                      contrastive_weight * contrastive_loss + 
                      attention_weight * attention_loss
                      )
                    #   density_weight * density_loss + 
                    #   contrastive_weight * contrastive_loss
                    #   attention_weight * attention_loss
                    #   )


        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
 
  
def validate_clam_lt(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):  
    model.eval()  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    inst_logger = Accuracy_Logger(n_classes=n_classes)  
    val_loss = 0.  
    val_error = 0.  
  
    val_inst_loss = 0.  
    val_inst_acc = 0.  
    inst_count=0  
      
    prob = np.zeros((len(loader), n_classes))  
    labels = np.zeros(len(loader))  
    sample_size = model.k_sample  
    with torch.inference_mode():  
        for batch_idx, (data, label, coords) in enumerate(loader):  # 注意这里接收三个值  
            data, label, coords = data.to(device), label.to(device), coords.to(device)        
            logits, Y_prob, Y_hat, _, instance_dict = model(data, coords=coords, label=label, instance_eval=True)  # 传递coords  
            acc_logger.log(Y_hat, label)  
              
            loss = loss_fn(logits, label)  
            val_loss += loss.item()  
  
            instance_loss = instance_dict['instance_loss']  
            inst_count+=1  
            instance_loss_value = instance_loss.item()  
            val_inst_loss += instance_loss_value  
  
            inst_preds = instance_dict['inst_preds']  
            inst_labels = instance_dict['inst_labels']  
            inst_logger.log_batch(inst_preds, inst_labels)  
  
            prob[batch_idx] = Y_prob.cpu().numpy()  
            labels[batch_idx] = label.item()  
              
            error = calculate_error(Y_hat, label)  
            val_error += error  
  
    # 其余逻辑与 validate_clam 相同 
    preds = np.argmax(prob, axis=1) 
    val_error /= len(loader)  
    val_loss /= len(loader)  
  
    if n_classes == 2:  
        auc = roc_auc_score(labels, prob[:, 1])
        f1 = f1_score(labels, preds)
        aucs = []  
    else:  
        aucs = []  
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])  
        for class_idx in range(n_classes):  
            if class_idx in labels:  
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])  
                aucs.append(calc_auc(fpr, tpr))  
            else:  
                aucs.append(float('nan'))  
        auc = np.nanmean(np.array(aucs))
        f1 = f1_score(labels, preds, average='macro')  
  
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {:.4f}'.format(val_loss, val_error, auc, f1)) 
    if inst_count > 0:  
        val_inst_loss /= inst_count  
        for i in range(2):  
            acc, correct, count = inst_logger.get_summary(i)  
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))  
      
    if writer:  
        writer.add_scalar('val/loss', val_loss, epoch)  
        writer.add_scalar('val/auc', auc, epoch)  
        writer.add_scalar('val/error', val_error, epoch)  
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        writer.add_scalar('val/f1', f1, epoch)  
  
    for i in range(n_classes):  
        acc, correct, count = acc_logger.get_summary(i)  
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  
          
        if writer and acc is not None:  
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)  
  
    if early_stopping:  
        assert results_dir  
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))  
          
        if early_stopping.early_stop:  
            print("Early stopping")  
            return True  
  
    return False
  
def summary_lt(model, loader, n_classes):  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    model.eval()  
    test_loss = 0.  
    test_error = 0.  
  
    all_probs = np.zeros((len(loader), n_classes))  
    all_labels = np.zeros(len(loader))  
    all_preds = np.zeros(len(loader))  
  
    slide_ids = loader.dataset.slide_data['slide_id']  
    patient_results = {}  
  
    for batch_idx, (data, label, coords) in enumerate(loader):  # 注意这里接收三个值  
        data, label, coords = data.to(device), label.to(device), coords.to(device)  
        slide_id = slide_ids.iloc[batch_idx]  
        with torch.inference_mode():  
            logits, Y_prob, Y_hat, A_raw, results_dict = model(  
                data,   
                coords=coords,  
                label=label,  # 添加label参数以支持instance_eval  
                instance_eval=True,  # 启用实例级评估  
                return_features=True  # 返回特征  
            ) 
  
        acc_logger.log(Y_hat, label)  
        probs = Y_prob.cpu().numpy()  
        all_probs[batch_idx] = probs  
        all_labels[batch_idx] = label.item()  
        all_preds[batch_idx] = Y_hat.item()  
          
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item(), 'results_dict': results_dict}})  
        error = calculate_error(Y_hat, label)  
        test_error += error  
  
    test_error /= len(loader)  
  
    if n_classes == 2:  
        auc = roc_auc_score(all_labels, all_probs[:, 1])  
        aucs = []  
        f1 = f1_score(all_labels, all_preds)  
    else:  
        aucs = []  
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])  
        for class_idx in range(n_classes):  
            if class_idx in all_labels:  
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])  
                aucs.append(calc_auc(fpr, tpr))  
            else:  
                aucs.append(float('nan'))  
  
        auc = np.nanmean(np.array(aucs))  
        f1 = f1_score(all_labels, all_preds, average='macro')  
  
    return patient_results, test_error, auc, f1, acc_logger

def compute_contrastive_loss(density_embeddings, labels, temperature=0.1):  
    """  
    计算 InfoNCE 对比损失  
    density_embeddings: (B, embed_dim) - batch 中每个 bag 的 density embedding  
    labels: (B,) - batch 中每个 bag 的标签  
    temperature: 温度参数  
    """  
    B = density_embeddings.shape[0]  
      
    # L2 归一化  
    density_embeddings = F.normalize(density_embeddings, dim=1)  
      
    # 计算相似度矩阵  
    similarity_matrix = torch.matmul(density_embeddings, density_embeddings.T) / temperature  # (B, B)  
      
    # 构建正样本 mask（同类样本）  
    labels = labels.contiguous().view(-1, 1)  
    mask_positive = torch.eq(labels, labels.T).float().to(density_embeddings.device)  # (B, B)  
      
    # 排除自己  
    mask_positive = mask_positive - torch.eye(B, device=density_embeddings.device)  
      
    # 计算 InfoNCE loss  
    exp_sim = torch.exp(similarity_matrix)  
      
    # 对每个样本计算损失  
    loss = 0.0  
    for i in range(B):  
        # 正样本的相似度之和  
        pos_sim = (exp_sim[i] * mask_positive[i]).sum()  
          
        # 所有样本的相似度之和（排除自己）  
        all_sim = exp_sim[i].sum() - exp_sim[i, i]  
          
        if pos_sim > 0:  # 只有当存在正样本时才计算  
            loss += -torch.log(pos_sim / (all_sim + 1e-8))  
      
    # 平均损失  
    num_positive_pairs = mask_positive.sum()  
    if num_positive_pairs > 0:  
        loss = loss / num_positive_pairs  
    else:  
        loss = torch.tensor(0.0, device=density_embeddings.device)  
      
    return loss


def compute_attention_entropy_loss(A, label, encourage_spread=True):  
    """  
    计算 attention entropy 正则化损失  
    A: attention weights (1, N)  
    label: bag label  
    encourage_spread: True 表示鼓励分散，False 表示鼓励集中  
    """  
    A = A.squeeze()  # (N,)  
      
    # 计算 entropy: -sum(a_i * log(a_i))  
    entropy = -(A * torch.log(A + 1e-8)).sum()  
      
    if encourage_spread:  
        # 鼓励分散：最大化 entropy（取负号变成最小化）  
        return -entropy  
    else:  
        # 鼓励集中：最小化 entropy  
        return entropy
    
def compute_attention_smoothing_loss(A, coords, sigma=100.0):  
    """  
    计算 attention local smoothing 损失  
    A: attention weights (1, N)  
    coords: patch coordinates (N, 2)  
    sigma: 高斯核的标准差  
    """  
    A = A.squeeze()  # (N,)  
    N = A.shape[0]  
      
    # 计算坐标距离矩阵  
    dist_matrix = torch.cdist(coords.float(), coords.float(), p=2)  # (N, N)  
      
    # 计算权重矩阵 w_ij = exp(-||p_i - p_j||^2 / sigma^2)  
    weights = torch.exp(-dist_matrix ** 2 / (sigma ** 2))  
      
    # 计算 smoothing loss: sum_i sum_j w_ij * (a_i - a_j)^2  
    A_diff = A.unsqueeze(1) - A.unsqueeze(0)  # (N, N)  
    smoothing_loss = (weights * A_diff ** 2).sum() / (N * N)  
      
    return smoothing_loss