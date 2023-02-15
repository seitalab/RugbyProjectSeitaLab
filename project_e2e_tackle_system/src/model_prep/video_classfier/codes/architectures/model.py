import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.video import r2plus1d_18, mc3_18, r3d_18

class Head(nn.Module):

    def __init__(self) -> None:
        super(Head, self).__init__()

        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(0.25)
    
    def forward(self, x: Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, num_classes).
        """
        feat = F.relu(self.fc1(x)) # -> (bs, 128)
        feat = self.drop1(feat)

        feat = self.fc2(feat) # -> (bs, 1)
        feat = torch.squeeze(feat, dim=-1)
        return feat

def prepare_model(modelname: str):
    """
    Args:

    Returns:

    """
    if modelname == "r2plus1d_18":
        backbone = r2plus1d_18(pretrained=True)
    elif modelname == "mc3_18":
        backbone = mc3_18(pretrained=True)
        # backbone = mc3_18(pretrained=False)
    elif modelname == "r3d_18":
        backbone = r3d_18(pretrained=True)
    else:
        raise NotImplementedError
    
    head = Head()
    model = Classifier(backbone, head)
    return model

class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, head: nn.Module):
        """
        Args:
            backbone
            head
        Returns:
            None
        """
        super(Classifier, self).__init__()
        self.backbone = backbone
        # for param in backbone.parameters():
        #     print(param)
        self.head = head

    def forward(self, x: Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        h = self.backbone(x) # (bs, params.backbone_out_dim)
        h = self.head(h) 
        return h