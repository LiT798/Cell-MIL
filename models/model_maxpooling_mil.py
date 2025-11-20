import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class MaxPooling_MIL(nn.Module):  
    def __init__(self, dropout=0., n_classes=2, embed_dim=1024):  
        super(MaxPooling_MIL, self).__init__()  
        self.n_classes = n_classes  
        self.embed_dim = embed_dim  
          
        self.classifier = nn.Sequential(  
            nn.Dropout(dropout),  
            nn.Linear(embed_dim, n_classes)  
        )  
      
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):  
        # Max pooling aggregation  
        M = torch.max(h, dim=0, keepdim=True)[0]  # [1, embed_dim]  
          
        logits = self.classifier(M)  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        Y_prob = F.softmax(logits, dim=1)  
          
        results_dict = {}  
        if return_features:  
            results_dict.update({'features': M})  
              
        return logits, Y_prob, Y_hat, None, results_dict