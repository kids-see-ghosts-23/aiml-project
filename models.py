import torch
import torch.nn as nn
import torchvision.models as models
from lightly.models.modules import SimCLRProjectionHead


def get_4channel_resnet(pretrained=True):
    """
    Creates a ResNet50 backbone modified to accept 4 input channels.
    Initializes the 4th channel using the mean of RGB weights.
    """
    # Load base model
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet50(weights=weights)

    # Modify the first layer (conv1) for 4 channels
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias,
    )

    with torch.no_grad():
        # Copy RGB weights to the first 3 channels
        new_conv1.weight[:, :3, :, :] = original_conv1.weight.data
        # Initialize 4th channel with the average of RGB
        new_conv1.weight[:, 3, :, :] = original_conv1.weight.data.mean(dim=1)

    model.conv1 = new_conv1
    # Remove the final classification layer for SSL
    model.fc = nn.Identity()
    return model


class SimCLRModel(nn.Module):
    """
    A SimCLR wrapper that attaches a projection head to the ResNet backbone.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # ResNet50 output is 2048. Head maps it to a 128-dim space for contrastive loss.
        self.projection_head = SimCLRProjectionHead(2048, 2048, 128)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(h)

