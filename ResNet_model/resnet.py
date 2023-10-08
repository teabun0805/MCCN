from torch import nn
from torchvision.models import resnet18


def get_resnet(pretrained: bool=True, num_classes: int=6) -> nn.Module:
    model = resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(
        in_channels=2,
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model