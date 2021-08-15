
import clip
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
from . import model_utils

from torchvision.transforms import Normalize


MODELS = {'RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'}

normalize_transform = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))


class ClipModel(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        model, _ = clip.load(model_name, device=self._device)
        self._model = model
        self._model.visual.float()
        self._classifier = None

    def forward(self, x):
        features = self.get_features(x)
        if self._classifier is None:
            return features
        return self._classifier(features)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val

    def new_last_layer(self, num_classes):
        num_in_features = self._model.visual.output_dim
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
        return self._model.encode_image(normalize_transform(x))
