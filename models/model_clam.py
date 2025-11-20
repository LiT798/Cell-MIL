import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        print(">>> forward input h type:", type(h))
        print(">>> forward input coords type:", type(coords))
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_LT(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,  
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,   
        use_pos_encoding=True, pos_dim=32,
        pos_type='linear', pos_fusion='concat',
        use_density=True, k_density=8,
        use_bag_statistics=True,
        config=None):  # 新增参数 use_pos_encoding , pos_dim 和 k_density ,config 参数（可选）
        super().__init__()
        # 参数优先由 config 控制（若提供），否则使用显式参数
        if config is not None:
            self.use_pos_encoding = config['use_pos_encoding']
            self.pos_type = config['pos_type']
            self.pos_fusion = config['pos_fusion']
            self.pos_dim = config['pos_dim']
            self.use_density = config['use_density']
            self.k_density = config['k_density']
            self.use_bag_statistics = config['use_bag_statistics']
        else:
            # 否则使用显式参数
            self.use_pos_encoding = use_pos_encoding
            self.pos_type = pos_type
            self.pos_fusion = pos_fusion
            self.pos_dim = pos_dim
            self.use_density = use_density
            self.k_density = k_density
            self.use_bag_statistics = use_bag_statistics

        # 原有的size_dict需要考虑位置编码维度  
        effective_embed_dim = embed_dim
        if self.use_pos_encoding:  
            effective_embed_dim = embed_dim + pos_dim  
        
        if self.use_density:  
            effective_embed_dim += 1  # 密度特征增加1维
              
        self.size_dict = {"small": [effective_embed_dim, 512, 256],   
                         "big": [effective_embed_dim, 512, 384]}

        # 添加位置编码器  
        if self.use_pos_encoding:
            self.pos_encoder = nn.Sequential(  
                nn.Linear(2, pos_dim // 2),  # 2D坐标 -> 中间维度  
                nn.ReLU(),  
                nn.Linear(pos_dim // 2, pos_dim),  # -> 最终位置编码维度  
                nn.Dropout(0.1)  
            )

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


        self.density_predictor = nn.Sequential(  
            nn.Linear(size[1], 128),  # size[1] 是 bag-level 特征维度  
            nn.ReLU(),  
            nn.Dropout(0.25),  
            nn.Linear(128, 5)  # 输出 5 个统计量  
        )

        # Density embedding 投影头（用于对比学习）  
        self.density_projector = nn.Sequential(  
            nn.Linear(5, 64),  # 5 个统计量 -> 64 维 embedding  
            nn.ReLU(),  
            nn.Linear(64, 128)  # 最终 128 维用于对比学习  
        )

        self.patch_density_estimator = PatchDensityEstimator(feat_dim=embed_dim, k=self.k_density, hidden_dim=64)

        
    def count_parameters(self, detailed: bool = True):
        """返回模型总参数数与各子模块参数数，detailed=True 时打印并返回 dict"""
        def _count(mod):
            if mod is None:
                return 0
            return sum(p.numel() for p in mod.parameters())

        totals = {}
        totals['total'] = sum(p.numel() for p in self.parameters())
        totals['attention_net'] = _count(self.attention_net)
        totals['classifiers'] = _count(self.classifiers)
        totals['instance_classifiers'] = sum(_count(m) for m in self.instance_classifiers)
        totals['pos_encoder'] = _count(getattr(self, 'pos_encoder', None))
        totals['density_predictor'] = _count(getattr(self, 'density_predictor', None))

        if detailed:
            print("Model parameter counts:")
            for k, v in totals.items():
                print(f"  {k}: {v:,}")
        return totals

    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, coords=None, label=None, instance_eval=False, return_features=False, attention_only=False):

        # ---------- [0] 类型检查 ----------
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"forward expected Tensor for features h, but got {type(h)}")
        # 保留原始未拼接位置/密度的特征，用于独立计算 bag-level 统计
        # clone 一份以防后续 in-place 修改影响原始特征
        h0 = h.clone()
        
        # ---------- [1] 可选位置编码 ----------
        if self.use_pos_encoding and coords is not None:  
            # 归一化坐标到[0,1]范围  
            # 假设coords是(N, 2)的tensor，包含(x, y)坐标  
            coords_float = coords.float()  # 确保是float类型  
            # 计算每个维度的最大值和最小值  
            coords_min = coords_float.min(dim=0, keepdim=True)[0]  
            coords_max = coords_float.max(dim=0, keepdim=True)[0]  

            normalized_coords = (coords_float - coords_min) / (coords_max - coords_min + 1e-8)  # 避免除以零
            
            # 根据 pos_type 选择不同位置编码方式
            if getattr(self, 'pos_type', 'linear') == 'linear':
                pos_encoding = self.pos_encoder(normalized_coords)  # 线性编码器
            elif self.pos_type == 'sin':
                pos_encoding = self.create_sinusoidal_pos_encoding(normalized_coords, self.pos_dim)
            else:
                raise ValueError(f"Unsupported pos_type: {self.pos_type}")

              # 拼接 or 相加
            if getattr(self, 'pos_fusion', 'concat') == 'concat':
                h = torch.cat([h, pos_encoding], dim=-1)
            elif self.pos_fusion == 'add':
                # 保证维度一致（必要时线性变换）
                if pos_encoding.shape[1] != h.shape[1]:
                    pos_encoding = F.linear(pos_encoding, torch.eye(pos_encoding.shape[1], h.shape[1], device=h.device))
                h = h + pos_encoding
            else:
                raise ValueError(f"Unsupported pos_fusion: {self.pos_fusion}")

        # ---------- [2] 可选局部密度 ----------
        if getattr(self, 'use_density', False):
            coords_float = coords.float()  # 确保是float类型
            density_features = self.patch_density_estimator(h, coords_float) 
            # density_features_old = self.compute_patch_density(h, k_density=self.k_density) #这里是消融实验用的无coords版本

            # 保证 dtype 一致
            if density_features.dtype != h.dtype:
                density_features = density_features.to(dtype=h.dtype)
            h = torch.cat([h, density_features], dim=-1)

        # ---------- [3] 主干注意力网络 ----------
        A, h = self.attention_net(h)  # NxK   注意力网络会自动适应新的输入维度     
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        # ---------- [4] 可选 bag-level 统计特征 ----------
        # 这里改为使用原始特征 h0（不包含 pos/density）计算统计量，避免 pos/density 对统计的直接干扰
        if getattr(self, 'use_bag_statistics', False):
            bag_stats = self.compute_bag_statistics(h0, A)
        else:
            bag_stats = None

        # ---------- [5] 分类预测 ----------  
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        # 预测统计特征
        bag_stats_true = bag_stats
        bag_stats_pred = None
        if bag_stats is not None and getattr(self, 'density_predictor', None) is not None:
            bag_stats_pred = self.density_predictor(M.squeeze())  # (5,)

         # ---------- [6] 可选 instance-level 评估 ----------
        results_dict = {}
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                    else:
                        continue
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
                
            })

        # ---------- [7] 其他输出 ----------
        if return_features:
            results_dict.update({'features': M})
            
        if bag_stats is not None:
            results_dict.update({
                'bag_stats_true': bag_stats_true,  
                'bag_stats_pred': bag_stats_pred,
                'bag_stats': bag_stats.detach().cpu(), # 不影响主干反传 
                'bag_embedding': M.squeeze().detach()  }) # 添加 bag embedding  
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def create_sinusoidal_pos_encoding(self, coords: torch.Tensor, pos_dim: int = 64):
        """
        生成二维坐标的正余弦位置编码 (sinusoidal positional encoding)

        参数：
            coords: (N, 2) 的张量，表示每个 patch 的归一化 (x, y) 坐标，范围 [0, 1]
            pos_dim: 输出编码维度（例如 32, 64）

        返回：
            pos_encoding: (N, pos_dim) 的张量，可直接拼接或相加到特征 h 上
        """
        assert coords.dim() == 2 and coords.size(1) == 2, "coords should be of shape (N, 2)"
        device = coords.device
        N = coords.size(0)

        # 将位置维度平均分配给 x 和 y 两个方向
        half_dim = pos_dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, device=device).float() * (-math.log(10000.0) / half_dim)
        )  # frequency scaling

        # 对 x 方向编码
        pos_x = coords[:, 0].unsqueeze(1)  # (N, 1)
        pos_x = pos_x * div_term  # 广播 (N, half_dim/2)
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=1)  # (N, half_dim)

        # 对 y 方向编码
        pos_y = coords[:, 1].unsqueeze(1)
        pos_y = pos_y * div_term
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=1)

        # 拼接 x 和 y 的编码
        pos_encoding = torch.cat([pos_x, pos_y], dim=1)  # (N, pos_dim 或 pos_dim-1)
        if pos_encoding.shape[1] < pos_dim:  # 若因奇数导致维度略小
            pos_encoding = F.pad(pos_encoding, (0, pos_dim - pos_encoding.shape[1]))

        return pos_encoding

    def compute_bag_statistics(self, h, A, logits_patches=None, k_density=8):  
        """  
        计算 bag-level 统计特征  
        h: patch embeddings (N, embed_dim)  
        A: attention weights (1, N)   
        logits_patches: patch-level logits (N, n_classes) - 可选  
        """  
        N = h.shape[0]  
        stats = []  
        
        # 1. Patch embedding 统计量  
        # Mean  
        embed_mean_scalar = torch.mean(h)  
        # Variance
        embed_var_scalar = torch.var(h) 

        # Skewness (简化实现)
        embed_centered = h - torch.mean(h, dim=0, keepdim=True)  
        embed_skew_scalar = torch.mean(torch.pow(embed_centered, 3)) / (torch.pow(torch.var(h) + 1e-8, 1.5) + 1e-8)
        
        stats.extend([embed_mean_scalar.unsqueeze(0),   
                  embed_var_scalar.unsqueeze(0),   
                  embed_skew_scalar.unsqueeze(0)])
         
        
        # 2. Top-k ratio (基于 attention weights)  
        top_k = max(5, int(0.1 * N))  
        A_flat = A.squeeze()  # (N,)  
        top_k_values, _ = torch.topk(A_flat, top_k)  
        top_k_attention_sum = torch.sum(top_k_values)  
        stats.append(top_k_attention_sum.unsqueeze(0)) 
        
        # 3. Local density estimation (基于 kNN)  
        if N > k_density:  
            # 计算 patch 间的距离矩阵  
            dist_matrix = torch.cdist(h, h, p=2)  # (N, N)  
            
            # 对每个 patch 找到 k 个最近邻  
            knn_distances, _ = torch.topk(dist_matrix, k_density + 1, largest=False, dim=1)  
            # 排除自己 (距离为0的点)  
            knn_distances = knn_distances[:, 1:]  # (N, k_density)  
            
            # 计算平均密度  
            avg_density = torch.mean(1.0 / (knn_distances.mean(dim=1) + 1e-8))  
            stats.append(avg_density.unsqueeze(0))  
        else:  
            # 如果 patch 数量太少，使用默认值  
            stats.append(torch.tensor([1.0], device=h.device))  
        
        return torch.cat(stats, dim=0)  # 拼接所有统计量
    
    def compute_patch_density(self, h, k_density=8):
        """
        计算 patch-level density 特征
        h: patch embeddings (N, embed_dim)
        return: density 特征向量 (N, 1)
        """

        N = h.shape[0]
        if N <= k_density:
            return torch.ones(N, 1, device=h.device)

        # 距离矩阵
        dist_matrix = torch.cdist(h, h, p=2)  # (N, N)

        # k 近邻
        knn_distances, _ = torch.topk(dist_matrix, k_density + 1, largest=False, dim=1)
        knn_distances = knn_distances[:, 1:]  # 去掉自己

        # 密度 = 邻居距离的倒数
        density = 1.0 / (knn_distances.mean(dim=1, keepdim=True) + 1e-8)  # (N, 1)

        return density

class PatchDensityEstimator(nn.Module):
    def __init__(self, feat_dim=1024, k=8, hidden_dim=64):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h0, coords):
        """
        h0: [N, D]  patch特征
        coords: [N, 2] patch坐标
        """
        N = h0.shape[0]
        
        # Step 1: 计算欧氏距离矩阵
        dist = torch.cdist(coords, coords)  # [N, N]
        knn_idx = dist.topk(k=self.k+1, largest=False).indices[:, 1:]  # [N, k]

        # Step 2: 获取邻域特征并计算相似度
        neighbor_feats = h0[knn_idx]                       # [N, k, D]
        h_i = h0.unsqueeze(1).expand_as(neighbor_feats)    # [N, k, D]
        sim = F.cosine_similarity(h_i, neighbor_feats, dim=-1)  # [N, k]

        # Step 3: 空间距离权重
        spatial_w = torch.exp(-dist.gather(1, knn_idx) / 0.05)  # [N, k]

        # Step 4: 聚合（取均值作为局部密度）
        sim_mean = sim.mean(dim=1, keepdim=True)
        spatial_mean = spatial_w.mean(dim=1, keepdim=True)

        # Step 5: 通过MLP融合
        density_input = torch.cat([sim_mean, spatial_mean], dim=1)  # [N, 2]
        density = self.mlp(density_input)  # [N, 1]
        
        return density