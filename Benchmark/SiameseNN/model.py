import torch
import torch.nn as nn
import torch.nn.functional as F

from torchviz import make_dot
from torchvision.models import swin_b, Swin_B_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=1024, proj_dim=128, freeze_backbone=False):
        super().__init__()
        
        # Swin-B can be any model swin_t, vgg and ect
        self.backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        # drop classification head
        self.backbone.head = nn.Identity()
        
        # Optional projection head
        self.proj_head = nn.Linear(embed_dim, proj_dim)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_once(self, x):
        """Forward pass of a single input (one branch)."""
        x = self.backbone(x)  # [batch_size, 1024]
        x = self.proj_head(x) # [batch_size, proj_dim]
        return x

    def forward(self, x1, x2):
        """Return the embeddings for both inputs."""
        emb_1 = self.forward_once(x1)
        emb_2 = self.forward_once(x2)
        return emb_1, emb_2


if __name__ == "__main__":
    siamese_model = SiameseNetwork()

    # Create dummy inputs (batch_size=1, 3 channels, 224x224)
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)

    # Forward pass: get the two embeddings
    emb_1, emb_2 = siamese_model(x1, x2)

    print(f"Embedding 1 shape {emb_1.shape}")
    print(f"Embedding 1 is {emb_1}")
    print(f"Embedding 2 shape {emb_2.shape}")
    print(f"Embedding 2 is {emb_2}")