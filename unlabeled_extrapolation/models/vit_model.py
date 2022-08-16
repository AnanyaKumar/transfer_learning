# Note: this model does normalization.
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
from . import model_utils

from torchvision.transforms import Normalize


MODELS = {'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'}


normalize_transform = Normalize(
    mean=(0.485, 0.456, 0.406),
    std=[0.228, 0.224, 0.225])


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


class VitModel(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        model = torch.hub.load('facebookresearch/dino:main', model_name)
        if self._device == 'cuda':
            model.cuda()
        self._model = model
        self._classifier = None

    def forward(self, x):
        features = self.get_features(x)
        if self._classifier is None:
            return features
        return self._classifier(features)

    def get_layers(self):
        patch_embed = self._model.patch_embed
        layers = [patch_embed, patch_embed]  # To streamline with CLIP ViT.
        layers += list(self._model.blocks)
        layers += [self._classifier]
        return layers

    def freeze_bottom_k(self, k):
        layers = self.get_layers()
        for i in range(min(k, len(layers))):
            set_requires_grad(layers[i], False)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val

    def new_last_layer(self, num_classes):
        num_in_features = self._model.norm.normalized_shape[0]
        self._classifier = nn.Linear(num_in_features, num_classes)
        self._classifier.to(self._device)

    def add_probe(self, probe):
        self._classifier = probe

    def get_last_layer(self):
        return self._classifier

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self._classifier, coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')

    def get_features(self, x):
        return self._model(normalize_transform(x))
