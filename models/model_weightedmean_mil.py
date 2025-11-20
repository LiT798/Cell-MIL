import torch  
import torch.nn as nn  
import torch.nn.functional as F

class WeightedMean_MIL(nn.Module):  
    def __init__(self, dropout=0., n_classes=2, embed_dim=1024):  
        super(WeightedMean_MIL, self).__init__()  
        self.n_classes = n_classes  
        self.embed_dim = embed_dim  
          
        # 权重网络  
        self.weight_net = nn.Sequential(  
            nn.Linear(embed_dim, 128),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(128, 1),  
            nn.Sigmoid()  
        )  
          
        self.classifier = nn.Sequential(  
            nn.Dropout(dropout),  
            nn.Linear(embed_dim, n_classes)  
        )  
      
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):  
        # 计算权重  
        weights = self.weight_net(h)  # [N, 1]  
        weights = weights / torch.sum(weights, dim=0, keepdim=True)  # 归一化  
          
        # 加权平均  
        M = torch.sum(h * weights, dim=0, keepdim=True)  # [1, embed_dim]  
          
        logits = self.classifier(M)  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        Y_prob = F.softmax(logits, dim=1)  
          
        results_dict = {}  
        if return_features:  
            results_dict.update({'features': M})  
              
        return logits, Y_prob, Y_hat, weights.squeeze(), results_dict