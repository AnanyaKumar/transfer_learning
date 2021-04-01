
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn

PRETRAIN_STYLE = ['supervised', 'mocov2', 'simclrv2']


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


class ResNet50(nn.Module):

    def __init__(self, pretrained=False, pretrain_style='supervised', checkpoint_path=None):
        super().__init__()
        if pretrain_style not in PRETRAIN_STYLE:
            raise ValueError(
                f'Pretrain style should be in {PRETRAIN_STYLE} but was {pretrain_style}')
        if pretrained and pretrain_style == 'mocov2':
            self._model = load_moco(checkpoint_path)
        elif pretrained and pretrain_style == 'simclrv2':
            raise NotImplementedError()
        elif not pretrained or pretrain_style == 'supervised':
            self._model = resnet50(pretrained=pretrained)

    def forward(self, x):
        return self._model(x)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
                param.requires_grad = val

    def new_last_layer(self, num_classes):
        num_in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_in_features, num_classes)

    def add_probe(self, probe):
        self._model.fc = probe

