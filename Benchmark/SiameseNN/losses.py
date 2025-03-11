import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss function from Hadsell et al. (2006).
    
    Args:
        margin (float): Non-negative margin value.
    
    Usage:
        loss_fn = ContrastiveLoss(margin=1.0)
        loss = loss_fn(embd_1, embd_2, label)
        
    Shape:
        - embd_1, embd_2: (batch_size, embed_dim)
        - label: (batch_size,) with entries in {0,1}.
        - Output: Scalar loss.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        if margin <= 0:
            raise ValueError("Margin must be a positive number.")
        self.margin = margin

    def forward(self, embd_1, embd_2, label):
        # Ensure label is float
        label = label.float()
        
        # Euclidean distance
        euclid_distance = torch.norm(embd_1 - embd_2, p=2, dim=1)
        
        # Contrastive loss
        loss_similar = label * euclid_distance.pow(2)
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclid_distance, min=0.0),
            2
        )
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss
