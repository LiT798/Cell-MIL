import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class Linear_MIL(nn.Module):  
    """  
    Linear MIL model with mean pooling aggregation  
    Args:  
        n_classes: number of output classes  
        embed_dim: input feature dimension (default: 1024)  
        dropout: dropout rate (default: 0.25)  
    """  
    def __init__(self, n_classes=2, embed_dim=1024, dropout=0.25):  
        super(Linear_MIL, self).__init__()  
        self.n_classes = n_classes  
        self.embed_dim = embed_dim  
          
        # Simple linear classifier  
        self.classifier = nn.Sequential(  
            nn.Linear(embed_dim, n_classes)  
        )  
          
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):  
        """  
        Forward pass  
        Args:  
            h: patch features (N, embed_dim)  
            label: ground truth label (not used in Linear_MIL)  
            instance_eval: whether to evaluate instance-level (not used)  
            return_features: whether to return aggregated features  
            attention_only: whether to return only attention (not applicable)  
        Returns:  
            logits: classification logits (1, n_classes)  
            Y_prob: class probabilities (1, n_classes)  
            Y_hat: predicted class (1, 1)  
            A_raw: attention weights (None for Linear_MIL)  
            results_dict: additional results dictionary  
        """  
        # Mean pooling aggregation  
        h = torch.mean(h, dim=0, keepdim=True)  # (1, embed_dim)  
          
        # Classification  
        logits = self.classifier(h)  # (1, n_classes)  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        Y_prob = F.softmax(logits, dim=1)  
          
        results_dict = {}  
        if return_features:  
            results_dict.update({'features': h})  
              
        # Return None for attention weights since Linear_MIL doesn't use attention  
        A_raw = None  
          
        return logits, Y_prob, Y_hat, A_raw, results_dict