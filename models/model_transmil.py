import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from nystrom_attention import NystromAttention  
  
class TransLayer(nn.Module):  
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):  
        super().__init__()  
        self.norm = norm_layer(dim)  
        self.attn = NystromAttention(  
            dim = dim,  
            dim_head = dim//8,  
            heads = 8,  
            num_landmarks = dim//2,  
            pinv_iterations = 6,  
            residual = True,  
            dropout=0.1  
        )  
  
    def forward(self, x):  
        x = x + self.attn(self.norm(x))  
        return x  
  
class PPEG(nn.Module):  
    def __init__(self, dim=512):  
        super(PPEG, self).__init__()  
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)  
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)  
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)  
  
    def forward(self, x, H, W):  
        B, _, C = x.shape  
        cls_token, feat_token = x[:, 0], x[:, 1:]  
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)  
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)  
        x = x.flatten(2).transpose(1, 2)  
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  
        return x  
  
class TransMIL(nn.Module):  
    def __init__(self, dropout=0., n_classes=2, embed_dim=1024, k_sample=8,   
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):  
        super(TransMIL, self).__init__()  
        self.n_classes = n_classes  
        self.embed_dim = embed_dim  
        self.k_sample = k_sample  
        self.instance_loss_fn = instance_loss_fn  
        self.subtyping = subtyping  
          
        # TransMIL components  
        self.pos_layer = PPEG(dim=512)  
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU())  
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))  
        self.layer1 = TransLayer(dim=512)  
        self.layer2 = TransLayer(dim=512)  
        self.norm = nn.LayerNorm(512)  
          
        # Classification head with dropout  
        self.classifier = nn.Sequential(  
            nn.Dropout(dropout),  
            nn.Linear(512, n_classes)  
        )  
          
        # CLAM-required components: Instance classifiers  
        instance_classifiers = [nn.Linear(512, 2) for i in range(n_classes)]  
        self.instance_classifiers = nn.ModuleList(instance_classifiers)  
          
        # Attention network for patch-level attention scores  
        self.attention_net = nn.Sequential(  
            nn.Linear(512, 256),  
            nn.Tanh(),  
            nn.Linear(256, 1)  
        )  
  
    @staticmethod  
    def create_positive_targets(length, device):  
        return torch.full((length, ), 1, device=device).long()  
      
    @staticmethod  
    def create_negative_targets(length, device):  
        return torch.full((length, ), 0, device=device).long()  
      
    def inst_eval(self, A, h, classifier):  
        """Instance-level evaluation for in-the-class attention branch"""  
        device = h.device  
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
      
    def inst_eval_out(self, A, h, classifier):  
        """Instance-level evaluation for out-of-the-class attention branch"""  
        device = h.device  
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
        # h shape: [n_patches, embed_dim] - CLAM format  
        original_h = h  # Keep original for attention computation  
          
        # Add batch dimension for TransMIL processing  
        if len(h.shape) == 2:  
            h = h.unsqueeze(0)  # [1, n_patches, embed_dim]  
          
        # TransMIL forward pass  
        h = self._fc1(h)  # [B, n, 512]  
          
        # Square padding logic  
        H = h.shape[1]  
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  
        add_length = _H * _W - H  
        h = torch.cat([h, h[:,:add_length,:]], dim=1)  # [B, N, 512]  
  
        # Add cls_token with improved device handling  
        B = h.shape[0]  
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)  
        h = torch.cat((cls_tokens, h), dim=1)  
  
        # Transformer layers  
        h = self.layer1(h)  # [B, N, 512]  
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]  
        h = self.layer2(h)  # [B, N, 512]  
  
        # Extract features for attention computation  
        patch_features = h[:, 1:H+1, :]  # Remove cls_token and padding  
        patch_features = patch_features.squeeze(0)  # [n_patches, 512]  
          
        # Compute attention scores for CLAM compatibility  
        A = self.attention_net(patch_features)  # [n_patches, 1]  
        A = A.transpose(1, 0)  # [1, n_patches] - CLAM format  
        A_raw = A.clone()  
          
        if attention_only:  
            return A  
          
        # Apply softmax to attention scores  
        A = F.softmax(A, dim=1)  # softmax over patches  
          
        # Instance-level evaluation (CLAM clustering constraints)  
        if instance_eval and label is not None:  
            total_inst_loss = 0.0  
            all_preds = []  
            all_targets = []  
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  
              
            for i in range(len(self.instance_classifiers)):  
                inst_label = inst_labels[i].item()  
                classifier = self.instance_classifiers[i]  
                if inst_label == 1:  # in-the-class  
                    instance_loss, preds, targets = self.inst_eval(A, patch_features, classifier)  
                    all_preds.extend(preds.cpu().numpy())  
                    all_targets.extend(targets.cpu().numpy())  
                else:  # out-of-the-class  
                    if self.subtyping:  
                        instance_loss, preds, targets = self.inst_eval_out(A, patch_features, classifier)  
                        all_preds.extend(preds.cpu().numpy())  
                        all_targets.extend(targets.cpu().numpy())  
                    else:  
                        continue  
                total_inst_loss += instance_loss  
  
            if self.subtyping:  
                total_inst_loss /= len(self.instance_classifiers)  
  
        # Aggregate features using attention weights  
        M = torch.mm(A, patch_features)  # [1, 512]  
          
        # Classification  
        logits = self.classifier(M)  # [1, n_classes]  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        Y_prob = F.softmax(logits, dim=1)  
          
        # Remove batch dimension to match CLAM format  
        # logits = logits.squeeze(0)  
        # Y_hat = Y_hat.squeeze(0)  
        # Y_prob = Y_prob.squeeze(0)  
          
        # Prepare results dictionary  
        results_dict = {}  
        if instance_eval and label is not None:  
            results_dict = {  
                'instance_loss': total_inst_loss,   
                'inst_labels': np.array(all_targets),   
                'inst_preds': np.array(all_preds)  
            }  
          
        if return_features:  
            results_dict.update({'features': M.squeeze(0)})  
          
        return logits, Y_prob, Y_hat, A_raw, results_dict