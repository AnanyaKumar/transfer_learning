
import torchvision
import torch
from torch import nn
from . import model_utils


class WildsVisionModel(nn.Module):

    def __init__(self, name, pretrained=False, checkpoint_path=None, num_classes=None):
        super().__init__()
        if name == 'wideresnet50':
            constructor_name = 'wide_resnet50_2'
            self._last_layer_name = 'fc'
        elif name == 'densenet121':
            constructor_name = name
            self._last_layer_name = 'classifier'
        elif name in ('resnet50', 'resnet34'):
            constructor_name = name
            self._last_layer_name = 'fc'
        else:
            raise ValueError(f'Torchvision model {name} not recognized')
        constructor = getattr(torchvision.models, constructor_name)
        # Create model. If pretrained and checkpoint_path is specified then we load a custom checkpoint
        # not pytorch's imagenet pretrained model.
        if checkpoint_path is not None and not(pretrained):
            raise ValueError('If supplying checkpoint_path, set pretrained=True')
        elif pretrained and checkpoint_path is None:
            self._model = constructor(pretrained=True)
        else:
            self._model = constructor()
        # If num_classes is specified, then initialize a new head for the model.
        if num_classes is not None:
            if type(num_classes) != int:
                raise ValueError('num_classes should be an int.')
            self.new_last_layer(num_classes)
        # If checkpoint is specified, then load model from checkpoint
        if checkpoint_path is not None:
            assert(pretrained)
            checkpoint = torch.load(checkpoint_path)
            self._model.load_state_dict(checkpoint['algorithm'], strict=False)

    def forward(self, x):
        return self._model(x)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
                param.requires_grad = val

    def new_last_layer(self, num_classes):
        num_in_features = getattr(self._model, self._last_layer_name).in_features
        last_layer = nn.Linear(num_in_features, num_classes)
        setattr(self._model, self._last_layer_name, last_layer)

    def add_probe(self, probe):
        setattr(self._model, self._last_layer_name, probe)

    def get_last_layer(self):
        return getattr(self._model, self._last_layer_name)
    
    def set_last_layer(self, coef, intercept):
        last_layer = getattr(self._model, self._last_layer_name) 
        model_utils.set_linear_layer(last_layer, coef, intercept)

    def get_feature_extractor(self):
        return nn.Sequential(*list(self._model.children())[:-1])

    def get_features(self, x):
        return self.get_feature_extractor()(x)

