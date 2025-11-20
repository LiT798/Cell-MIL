import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class DSMIL_MIL(nn.Module):  
    def __init__(self, dropout=0., n_classes=2, embed_dim=1024, dropout_v=0.0, nonlinear=True, passing_v=False):  
        super(DSMIL_MIL, self).__init__()  
        self.n_classes = n_classes  
        self.embed_dim = embed_dim  
          
        # Instance-level classifier (IClassifier equivalent)  
        self.i_classifier = nn.Sequential(  
            nn.Dropout(dropout),  
            nn.Linear(embed_dim, n_classes)  
        )  
          
        # Bag-level classifier (BClassifier equivalent)  
        self.b_classifier = BClassifier(  
            input_size=embed_dim,   
            output_class=n_classes,   
            dropout_v=dropout_v,   
            nonlinear=nonlinear,   
            passing_v=passing_v  
        )  
          
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):  
        # h: [N, embed_dim] - N instances with embed_dim features each  
          
        # Instance-level predictions  
        instance_logits = self.i_classifier(h)  # [N, n_classes]  
          
        # Bag-level aggregation using attention mechanism  
        bag_logits, attention_weights, bag_features = self.b_classifier(h, instance_logits)  
          
        # Final predictions (using bag-level logits as primary)  
        Y_prob = F.softmax(bag_logits, dim=1)  
        Y_hat = torch.topk(bag_logits, 1, dim=1)[1]  
          
        results_dict = {}  
        if return_features:  
            results_dict.update({'features': bag_features.squeeze()})  
        if attention_only or instance_eval:  
            results_dict.update({'attention_weights': attention_weights})  
            results_dict.update({'instance_logits': instance_logits})  
              
        return bag_logits, Y_prob, Y_hat, attention_weights, results_dict

  
class BClassifier(nn.Module):  
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  
        super(BClassifier, self).__init__()  
          
        # Query network  
        if nonlinear:  
            self.q = nn.Sequential(  
                nn.Linear(input_size, 128),   
                nn.ReLU(),   
                nn.Linear(128, 128),   
                nn.Tanh()  
            )  
        else:  
            self.q = nn.Linear(input_size, 128)  
              
        # Value network  
        if passing_v:  
            self.v = nn.Sequential(  
                nn.Dropout(dropout_v),  
                nn.Linear(input_size, input_size),  
                nn.ReLU()  
            )  
        else:  
            self.v = nn.Identity()  
          
        # Final classifier (1D convolution)  
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
          
    def forward(self, feats, c):  
        device = feats.device  
        V = self.v(feats)  # [N, input_size]  
        Q = self.q(feats)  # [N, 128]  
          
        # Find critical instances by sorting class scores  
        _, m_indices = torch.sort(c, 0, descending=True)  # [N, output_class]  
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # [output_class, input_size]  
        q_max = self.q(m_feats)  # [output_class, 128]  
          
        # Compute attention weights  
        A = torch.mm(Q, q_max.transpose(0, 1))  # [N, output_class]  
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  
          
        # Aggregate features using attention  
        B = torch.mm(A.transpose(0, 1), V)  # [output_class, input_size]  
          
        # Final classification  
        B = B.view(1, B.shape[0], B.shape[1])  # [1, output_class, input_size]  
        C = self.fcc(B)  # [1, output_class, 1]  
        C = C.view(1, -1)  # [1, output_class]  
          
        return C, A, B