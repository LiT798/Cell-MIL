from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits, Generic_MIL_Dataset_WithCoords
from models.model_clam import CLAM_LT
from utils.clamlt_core_utils import summary_lt
from utils.utils import get_split_loader_with_coords
from sklearn.metrics import auc as calc_auc 
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'clam_lt', 'mil', 'maxpooling_mil', 'meanpooling_mil', 'linear_mil', 'weightedmean_mil', 'dsmil', 'transmil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_3_TCGA_Lung', 'task_4_TCGA_BRCA', 'task_5_EndoScell', 'task_6_EndoScell'], required=True,)
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--pos_dim', type=int, default=32,  
                   help='positional encoding dimension')
parser.add_argument('--use_pos_encoding', action='store_true', default=False,
                    help='Enable positional encoding module (default: False)')
parser.add_argument('--use_density', action='store_true', default=False,
                    help='Enable density feature module (default: False)')
parser.add_argument('--use_bag_statistics', action='store_true', default=False,
                    help='Enable bag-level statistics module (default: False)')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'task_3_TCGA_Lung':
    args.n_classes=2
    dataset = Generic_MIL_Dataset_WithCoords(csv_path = 'dataset_csv/DATA_TCGA_Lung.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'TCGA-LUAD':0, 'TCGA-LUSC':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_4_TCGA_BRCA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset_WithCoords(csv_path = 'dataset_csv/TCGA_BRCA.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_5_EndoScell':
    args.n_classes=2
    dataset = Generic_MIL_Dataset_WithCoords(csv_path = 'dataset_csv/EndoScell.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'negative':0, 'positive':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'task_6_EndoScell':
    args.n_classes=2
    dataset = Generic_MIL_Dataset_WithCoords(csv_path = 'dataset_csv/endoscell_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'negative':0, 'positive':1},
                            patient_strat=False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}


def summary_lt_eval(model, loader, n_classes):
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
                label=label,  # 添加label参数  
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

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
  
    return patient_results, test_error, auc, df, f1, acc_logger


def eval_lt(dataset, args, ckpt_path):    
    # 初始化模型    
    model_dict = {    
        "dropout": args.drop_out,     
        'n_classes': args.n_classes,     
        "embed_dim": args.embed_dim,    
        "use_pos_encoding": args.use_pos_encoding,    
        "pos_dim": args.pos_dim    
    }    
        
    if args.model_size is not None:    
        model_dict.update({"size_arg": args.model_size})    
        
    model = CLAM_LT(**model_dict)    
        
    # 加载检查点并清理不需要的键  
    ckpt = torch.load(ckpt_path)  
    ckpt_clean = {}  
    for key in ckpt.keys():  
        # 跳过 instance_loss_fn 相关的键  
        if 'instance_loss_fn' in key:  
            continue  
        # 移除 .module 前缀（如果存在，用于多GPU训练）  
        ckpt_clean.update({key.replace('.module', ''): ckpt[key]})  
      
    model.load_state_dict(ckpt_clean, strict=True)  
    model = model.to(device)    
    model.eval()    
        
    # 创建数据加载器（使用支持坐标的版本）    
    loader = get_split_loader_with_coords(dataset)    
        
    # 使用 summary_lt 进行评估    
    patient_results, test_error, auc, df, f1, acc_logger = summary_lt_eval(model, loader, args.n_classes)    
        
    print('test_error: ', test_error)    
    print('auc: ', auc)    
        
    return model, patient_results, test_error, auc, df, f1


if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []  # 添加F1列表
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df, f1 = eval_lt(split_dataset, args, ckpt_paths[ckpt_idx])  
        all_f1.append(f1)  # 收集F1分数 
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_f1': all_f1})
    # 计算均值和标准差  
    mean_test_auc = np.mean(all_auc)  
    std_test_auc = np.std(all_auc)  
    mean_test_acc = np.mean(all_acc)  
    std_test_acc = np.std(all_acc)  
    mean_test_f1 = np.mean(all_f1)  
    std_test_f1 = np.std(all_f1)  

    # 添加汇总行  
    summary_row = pd.DataFrame({  
        'folds': ['mean±std'],  
        'test_auc': [f'{mean_test_auc:.4f}±{std_test_auc:.4f}'],  
        'val_auc': [''],  # 如果不需要val指标的汇总可以留空  
        'test_acc': [f'{mean_test_acc:.4f}±{std_test_acc:.4f}'],  
        'val_acc': [''],  
        'test_f1': [f'{mean_test_f1:.4f}±{std_test_f1:.4f}']  
    })  
    
    final_df = pd.concat([final_df, summary_row], ignore_index=True)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
