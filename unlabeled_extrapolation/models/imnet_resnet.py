
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
from . import model_utils
from torchvision.transforms import Normalize

import unlabeled_extrapolation.utils.utils as utils

try:
    from models import swav_resnet50
except:
    from . import swav_resnet50


PRETRAIN_STYLE = ['supervised', 'mocov2', 'swav', 'simclrv2']

imnet_normalize_transform = Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.228, 0.224, 0.225))


def load_moco(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.__dict__[checkpoint['arch']]()

    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    return model


def load_swav(checkpoint_path):
    model = swav_resnet50.resnet50(output_dim=0, eval_mode=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # remove prefix "module."
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    for k, v in model.state_dict().items():
        if k not in list(state_dict):
            print('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            print('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v
    msg = model.load_state_dict(state_dict, strict=False)
    model = torch.nn.Sequential(OrderedDict([
        ('resnet', model),
        ('fc', nn.Linear(2048, 1000))]))
    return model


class ResNet50(nn.Module):

    def __init__(self, pretrained=False, pretrain_style='supervised', checkpoint_path=None, normalize=False):
        super().__init__()
        if pretrain_style not in PRETRAIN_STYLE:
            raise ValueError(
                f'Pretrain style should be in {PRETRAIN_STYLE} but was {pretrain_style}')
        if pretrained and pretrain_style == 'mocov2':
            self._model = load_moco(checkpoint_path)
        if pretrained and pretrain_style == 'swav':
            self._model = load_swav(checkpoint_path)
        elif pretrained and pretrain_style == 'simclrv2':
            raise NotImplementedError()
        elif not pretrained or pretrain_style == 'supervised':
            self._model = resnet50(pretrained=pretrained)
        self._side_tuning = False
        self._normalize = normalize

    def forward(self, x):
        if self._side_tuning:
            pretrained_reps = self.get_features(x)
            side_reps = self._side_network.get_features(x)
            print(pretrained_reps.shape, side_reps.shape)
            return self._model.fc(pretrained_reps + side_reps)
        return self._model(x)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
                param.requires_grad = val

    def enable_side_tuning(self):
        self._side_tuning = True
        # Freeze gradients for the original network.
        self.set_requires_grad(val=False)
        # Create the last layer again, so that parameters are not frozen.
        self.new_last_layer(num_classes=self._model.fc.out_features)
        # Add a side network, as per the side-tuning paper.
        # In many of their experiments, this is also a ResNet-50 like the original network.
        self._side_network = ResNet50()

    def get_layers(self):
        stem = nn.ModuleList([self._model.conv1, self._model.bn1, self._model.relu, self._model.maxpool])
        layers = ([('stem', stem)] +
                  [('layer1', self._model.layer1)] +
                  [('layer2', self._model.layer2)] +
                  [('layer3', self._model.layer3)] +
                  [('layer4', self._model.layer4)] +
                  [('avgpool', self._model.avgpool)] +
                  [('head', self._model.fc)])
        return layers

    def freeze_bottom_k(self, k):
        layers = [l for _, l in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        for i in range(k):
            utils.set_requires_grad(layers[i], False)

    def new_last_layer(self, num_classes):
        num_in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_in_features, num_classes)

    def add_probe(self, probe):
        self._model.fc = probe

    def get_last_layer(self):
        return self._model.fc

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self._model.fc, coef, intercept) 
    
    def get_feature_extractor(self):
        return nn.Sequential(*list(self._model.children())[:-1])

    def get_features(self, x):
        # Note: For side-tuning, this gives the original pretrained features.
        # It's a big ambiguous what the 'features' are in that case.
        if self._normalize:
            x = imnet_normalize_transform(x) 
        features = self.get_feature_extractor()(x)
        return torch.reshape(features, (features.shape[0], -1))

