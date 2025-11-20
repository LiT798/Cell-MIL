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

def train_loop_transmil(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):  
    """TransMIL专用训练循环，使用标准的单一损失机制"""  
    model.train()  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
      
    train_loss = 0.  
    train_error = 0.  
  
    print('\n')  
    for batch_idx, (data, label) in enumerate(loader):  
        data, label = data.to(device), label.to(device)  
          
        # TransMIL模型前向传播  
        logits, Y_prob, Y_hat, attention_weights, results_dict = model(data, label=label)  
          
        acc_logger.log(Y_hat, label)  
          
        # TransMIL使用标准交叉熵损失  
        loss = loss_fn(logits, label)  
          
        train_loss += loss.item()  
          
        if (batch_idx + 1) % 20 == 0:  
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(  
                batch_idx, loss.item(), label.item(), data.size(0)))  
  
        error = calculate_error(Y_hat, label)  
        train_error += error  
          
        # 反向传播  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
  
    # 计算epoch的损失和错误率  
    train_loss /= len(loader)  
    train_error /= len(loader)  
  
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(  
        epoch, train_loss, train_error))  
      
    for i in range(n_classes):  
        acc, correct, count = acc_logger.get_summary(i)  
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  
        if writer and acc is not None:  
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)  
  
    if writer:  
        writer.add_scalar('train/loss', train_loss, epoch)  
        writer.add_scalar('train/error', train_error, epoch)  
  
def validate_transmil(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):  
    """TransMIL验证函数，处理TransMIL模型的验证逻辑"""  
    model.eval()  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    val_loss = 0.  
    val_error = 0.  
      
    prob = np.zeros((len(loader), n_classes))  
    labels = np.zeros(len(loader))  
      
    with torch.inference_mode():  
        for batch_idx, (data, label) in enumerate(loader):  
            data, label = data.to(device), label.to(device)  
              
            logits, Y_prob, Y_hat, attention_weights, results_dict = model(data, label=label)  
            acc_logger.log(Y_hat, label)  
              
            # TransMIL使用标准损失  
            loss = loss_fn(logits, label)  
            val_loss += loss.item()  
  
            prob[batch_idx] = Y_prob.cpu().numpy()  
            labels[batch_idx] = label.item()  
              
            error = calculate_error(Y_hat, label)  
            val_error += error  
  
    val_error /= len(loader)  
    val_loss /= len(loader)  
  
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
  
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(  
        val_loss, val_error, auc))  
      
    if writer:  
        writer.add_scalar('val/loss', val_loss, epoch)  
        writer.add_scalar('val/auc', auc, epoch)  
        writer.add_scalar('val/error', val_error, epoch)  
  
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
  
def summary_transmil(model, loader, n_classes):  
    """TransMIL评估函数，专门处理TransMIL模型的最终评估"""  
    acc_logger = Accuracy_Logger(n_classes=n_classes)  
    model.eval()  
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
            logits, Y_prob, Y_hat, attention_weights, results_dict = model(data)  
  
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
                'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None  
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