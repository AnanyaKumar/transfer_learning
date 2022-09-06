# Note: this model does normalization.
# Also supports convnext and potentially some other timm models. TODO: refactor.

from collections import OrderedDict
import torchvision.models as models
import torch
from torch import nn
from . import model_utils

import timm
from timm.models import vision_transformer
from timm.models import convnext
from timm.data import resolve_data_config

from torchvision.transforms import Normalize

def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


def get_timm_transform(model_name):
    config = resolve_data_config({}, model=model_name)
    normalize_transform = Normalize(mean=config['mean'], std=config['std'])
    return normalize_transform


class TimmModel(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self._model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(model_name, pretrained=True)
        self._normalize_transform = get_timm_transform(model_name)
        if self._device == 'cuda':
            model.cuda()
        self._model = model

    def forward(self, x):
        return self._model(self._normalize_transform(x)) 

    def get_layers(self):
        if 'vit' in self._model_name:
            pos_embed = self._model.pos_embed
            cls_token = self._model.cls_token
            patch_embed = self._model.patch_embed
            blocks = self._model.blocks
            layers = [
                ('patch_embed', patch_embed),
                ('empty_ln_pre', nn.Module()),  # To streamline number of layers with CLIP.
                ('pos_embed', model_utils.ParamWrapperModule(pos_embed)),
                ('cls_token', model_utils.ParamWrapperModule(cls_token)),
            ] 
            for i, block in zip(range(len(blocks)), blocks):
                layers += [
                    ('trans' + str(i) + '_norm1', block.norm1),
                    ('trans' + str(i) + '_attn', block.attn),
                    ('trans' + str(i) + '_norm2', block.norm2),
                    ('trans' + str(i) + '_mlp', block.mlp),
                ]
            layers += [('post_norm', self._model.norm)]
            layers += [('head', self.get_last_layer())]
        elif 'convnext' in self._model_name:
            layers = [('stem', self._model.stem)]
            blocks = self._model.stages
            for i, block in zip(range(len(blocks)), blocks):
                layers += [('stage' + str(i), block)]
            # The convnext head contains multiple pieces. I checked that all the parameters
            # are accounted for by these two components (norm and fc).
            layers += [('post_norm', self._model.head.norm)]
            layers += [('head', self._model.head.fc)] 
        else:
            raise NotImplementedError('Model currently not implemented') 
        return layers

    def tune_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        set_requires_grad(self._model, False)
        for i in range(k):
            set_requires_grad(layers[i], True)
        # Also need to tune the prediction head because the fine-tuning task is different
        # from pretraining.
        set_requires_grad(layers[-1], True)
    
    def freeze_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        for i in range(min(k, len(layers))):
            set_requires_grad(layers[i], False)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val

    def new_last_layer(self, num_classes):
        if 'vit' in self._model_name:
            num_in_features = self._model.norm.normalized_shape[0]
            self._model.head = nn.Linear(num_in_features, num_classes)
        elif 'convnext' in self._model_name:
            num_in_features = self._model.head.norm.normalized_shape[0]
            self._model.head.fc = nn.Linear(num_in_features, num_classes)
        else:
            raise NotImplementedError('Model currently not implemented') 
        self._model.head.to(self._device)

    def add_probe(self, probe):
        if 'vit' in self._model_name:
            self._model.head = probe
        elif 'convnext' in self._model_name:
            self._model.head.fc = probe
        else:
            raise NotImplementedError('Model currently not implemented') 

    def get_last_layer(self):
        if 'vit' in self._model_name:
            return self._model.head
        elif 'convnext' in self._model_name:
            return self._model.head.fc
        else:
            raise NotImplementedError('Model currently not implemented') 

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self.get_last_layer(), coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')

    def get_features(self, x):
        raise NotImplementedError('Need to implement this for timm models.') 
