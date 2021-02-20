
from torchvision.models import resnet50
import torch
from torch import nn

class ResNet50(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self._model = resnet50(pretrained=pretrained)

    def forward(self, x):
        return self._model(x)

    def new_last_layer(self, num_classes):
        num_in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_in_features, num_classes)

