import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from utils.utils import *   
import os  
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc  
from sklearn.preprocessing import label_binarize  
  
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
  
def train_loop_dsmil(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):  
    """DSMIL专用训练循环，实现双重损失机制"""  
    model.train()  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
      
    train_loss = 0.  
    train_error = 0.  
    train_bag_loss = 0.  
    train_max_loss = 0.  
  
    print('\n')  
    for batch_idx, (data, label) in enumerate(loader):  
        data, label = data.to(device), label.to(device)  
          
        # DSMIL模型前向传播  
        bag_logits, Y_prob, Y_hat, attention_weights, results_dict = model(data, label=label, instance_eval=True)  
          
        acc_logger.log(Y_hat, label)  
          
        # 获取实例级预测  
        instance_logits = results_dict.get('instance_logits')  
          
        # 计算双重损失  
        bag_loss = loss_fn(bag_logits, label)  
        max_loss = loss_fn(torch.max(instance_logits, 0)[0].unsqueeze(0), label)  
          
        # 使用bag_weight参数控制损失权重  
        total_loss = bag_weight * bag_loss + (1 - bag_weight) * max_loss  
          
        bag_loss_value = bag_loss.item()  
        max_loss_value = max_loss.item()  
        train_bag_loss += bag_loss_value  
        train_max_loss += max_loss_value  
          
        train_loss += total_loss.item()  
          
        if (batch_idx + 1) % 20 == 0:  
            print('batch {}, bag_loss: {:.4f}, max_loss: {:.4f}, total_loss: {:.4f}, '.format(  
                batch_idx, bag_loss_value, max_loss_value, total_loss.item()) +   
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))  
  
        error = calculate_error(Y_hat, label)  
        train_error += error  
          
        # 反向传播  
        total_loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
  
    # 计算epoch的损失和错误率  
    train_loss /= len(loader)  
    train_error /= len(loader)  
    train_bag_loss /= len(loader)  
    train_max_loss /= len(loader)  
  
    print('Epoch: {}, train_loss: {:.4f}, train_bag_loss: {:.4f}, train_max_loss: {:.4f}, train_error: {:.4f}'.format(  
        epoch, train_loss, train_bag_loss, train_max_loss, train_error))  
      
    for i in range(n_classes):  
        acc, correct, count = acc_logger.get_summary(i)  
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  
        if writer and acc is not None:  
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)  
  
    if writer:  
        writer.add_scalar('train/loss', train_loss, epoch)  
        writer.add_scalar('train/error', train_error, epoch)  
        writer.add_scalar('train/bag_loss', train_bag_loss, epoch)  
        writer.add_scalar('train/max_loss', train_max_loss, epoch)  
  
def validate_dsmil(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):  
    """DSMIL验证函数，处理DSMIL模型的验证逻辑"""  
    model.eval()  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    val_loss = 0.  
    val_error = 0.  
    val_bag_loss = 0.  
    val_max_loss = 0.  
      
    prob = np.zeros((len(loader), n_classes))  
    labels = np.zeros(len(loader))  
      
    with torch.inference_mode():  
        for batch_idx, (data, label) in enumerate(loader):  
            data, label = data.to(device), label.to(device)  
              
            bag_logits, Y_prob, Y_hat, attention_weights, results_dict = model(data, label=label, instance_eval=True)  
            acc_logger.log(Y_hat, label)  
              
            # 获取实例级预测  
            instance_logits = results_dict.get('instance_logits')  
              
            # 计算双重损失  
            bag_loss = loss_fn(bag_logits, label)  
            max_loss = loss_fn(torch.max(instance_logits, 0)[0].unsqueeze(0), label)  
            total_loss = 0.5 * bag_loss + 0.5 * max_loss  
              
            val_loss += total_loss.item()  
            val_bag_loss += bag_loss.item()  
            val_max_loss += max_loss.item()  
  
            prob[batch_idx] = Y_prob.cpu().numpy()  
            labels[batch_idx] = label.item()  
              
            error = calculate_error(Y_hat, label)  
            val_error += error  
  
    val_error /= len(loader)  
    val_loss /= len(loader)  
    val_bag_loss /= len(loader)  
    val_max_loss /= len(loader)  
  
    if n_classes == 2:  
        auc = roc_auc_score(labels, prob[:, 1])  
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
  
    print('\nVal Set, val_loss: {:.4f}, val_bag_loss: {:.4f}, val_max_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(  
        val_loss, val_bag_loss, val_max_loss, val_error, auc))  
      
    if writer:  
        writer.add_scalar('val/loss', val_loss, epoch)  
        writer.add_scalar('val/auc', auc, epoch)  
        writer.add_scalar('val/error', val_error, epoch)  
        writer.add_scalar('val/bag_loss', val_bag_loss, epoch)  
        writer.add_scalar('val/max_loss', val_max_loss, epoch)  
  
    for i in range(n_classes):  
        acc, correct, count = acc_logger.get_summary(i)  
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  
          
        if writer and acc is not None:  
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)  
  
    if early_stopping:  
        assert results_dir  
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))  
          
        if early_stopping.early_stop:  
            print("Early stopping")  
            return True  
  
    return False  
  
def summary_dsmil(model, loader, n_classes):  
    """DSMIL评估函数，专门处理DSMIL模型的最终评估"""  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    model.eval()  
    test_loss = 0.  
    test_error = 0.  
  
    all_probs = np.zeros((len(loader), n_classes))  
    all_labels = np.zeros(len(loader)) 
    all_preds = np.zeros(len(loader))  # 添加这行  
  
    slide_ids = loader.dataset.slide_data['slide_id']  
    patient_results = {}  
  
    for batch_idx, (data, label) in enumerate(loader):  
        data, label = data.to(device), label.to(device)  
        slide_id = slide_ids.iloc[batch_idx]  
          
        with torch.inference_mode():  
            bag_logits, Y_prob, Y_hat, attention_weights, results_dict = model(data)  
  
        acc_logger.log(Y_hat, label)  
        probs = Y_prob.cpu().numpy()  
        all_probs[batch_idx] = probs  
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()  # 添加这行 
          
        # 保存注意力权重到结果中  
        patient_results.update({  
            slide_id: {  
                'slide_id': np.array(slide_id),   
                'prob': probs,   
                'label': label.item(),  
                'attention_weights': attention_weights.cpu().numpy()  
            }  
        })  
          
        error = calculate_error(Y_hat, label)  
        test_error += error  
  
    test_error /= len(loader)  
  
    if n_classes == 2:  
        auc = roc_auc_score(all_labels, all_probs[:, 1])  
        aucs = []
        f1 = f1_score(all_labels, all_preds)  # 二分类F1
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
        f1 = f1_score(all_labels, all_preds, average='macro')  # 宏平均F1 
  
    return patient_results, test_error, auc, f1, acc_logger