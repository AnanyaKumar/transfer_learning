# Note: this model does normalization.
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
from . import model_utils

import timm
from timm.models import vision_transformer
from timm.data import resolve_data_config

from torchvision.transforms import Normalize


MODELS = {
    'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8',
    'deit_base_patch16_224',
}


default_imnet_normalize = Normalize(
    mean=(0.485, 0.456, 0.406),
    std=[0.229, 0.224, 0.225])


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


def is_timm_vit_name(model_name):
    timm_vit_names = vision_transformer.default_cfgs.keys()
    return model_name.startswith('timm.') and get_timm_name(model_name) in timm_vit_names


def get_timm_name(model_name):
    assert model_name.startswith('timm.')
    return model_name[5:]


def get_timm_transform(model_name):
    timm_name = get_timm_name(model_name)
    config = resolve_data_config({}, model=timm_name)
    normalize_transform = Normalize(mean=config['mean'], std=config['std'])
    return normalize_transform


class VitModel(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        if model_name not in MODELS and not(is_timm_vit_name(model_name)):
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        if is_timm_vit_name(model_name):
            model = timm.create_model(get_timm_name(model_name), pretrained=True)
            self._normalize_transform = get_timm_transform(model_name)
        elif 'dino' in model_name:
            model = torch.hub.load('facebookresearch/dino:main', model_name, force_reload=True)
        elif 'deit' in model_name:
            model = torch.hub.load('facebookresearch/deit:main', model_name, force_reload=True)
        if self._device == 'cuda':
            model.cuda()
        if 'deit' in model_name:
            # deit implementation is not complete, doesn't seem to load properly, and also
            # get_layers likely isn't handled correctly.
            raise NotImplementedError
            self._model = nn.Sequential(*list(model.children())[:-1])
        else:
            self._model = model
        self._classifier = None

    def forward(self, x):
        if is_timm_vit_name(self._model_name):
            return self._model(self._normalize_transform(x)) 
        features = self.get_features(x)
        if self._classifier is None:
            return features
        return self._classifier(features)

    def get_layers(self):
        if 'deit' in self._model_name:
            patch_embed = self._model[0]
            blocks = self._model[2]
        else:
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
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val

    def new_last_layer(self, num_classes):
        if 'deit' in self._model_name:
            num_in_features = self._model[3].normalized_shape[0]
        else:
            num_in_features = self._model.norm.normalized_shape[0]
        if is_timm_vit_name(self._model_name):
            self._model.head = nn.Linear(num_in_features, num_classes)
            self._model.head.to(self._device)
        else:
            self._classifier = nn.Linear(num_in_features, num_classes)
            self._classifier.to(self._device)

    def add_probe(self, probe):
        if is_timm_vit_name(self._model_name):
            self._model.head = probe
        else:
            self._classifier = probe

    def get_last_layer(self):
        if is_timm_vit_name(self._model_name):
            return self._model.head
        else:
            return self._classifier

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self.get_last_layer(), coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')

    def get_features(self, x):
        # The normalize transform for Deit is standard imagenet transform. See:
        # https://github.com/facebookresearch/deit/blob/main/datasets.py which calls
        # the timm library create_transform, which uses these imagenet defaults.
        if is_timm_vit_name(self._model_name):
            raise NotImplementedError('Need to implement this for timm models.') 
        else:
            return self._model(default_imnet_normalize(x))
