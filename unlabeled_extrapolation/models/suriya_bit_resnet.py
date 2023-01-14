# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from torchvision.transforms import Normalize

import unlabeled_extrapolation.utils.utils as utils


class GroupNormPartialWrapper(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super(GroupNormPartialWrapper, self).__init__(num_groups, num_channels, eps=eps, affine=affine)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False, weight_standardization: bool = False):
    if not weight_standardization:
        return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)
    else:
        return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False, weight_standardization: bool = False):
    if not weight_standardization:
        return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1, norm_layer = GroupNormPartialWrapper):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4
        weight_standardization = issubclass(norm_layer,nn.GroupNorm)

        self.gn1 = norm_layer(cin)
        self.conv1 = conv1x1(cin, cmid, weight_standardization=weight_standardization)
        self.gn2 = norm_layer(cmid)
        self.conv2 = conv3x3(cmid, cmid, stride, weight_standardization=weight_standardization)  # Original code has it on conv1!!
        self.gn3 = norm_layer(cmid)
        self.conv3 = conv1x1(cmid, cout, weight_standardization=weight_standardization)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, weight_standardization=weight_standardization)
    
    def forward(self, x):
        out = self.relu(self.gn1(x))
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)
        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False, norm_layer = GroupNormPartialWrapper):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        if issubclass(norm_layer, nn.GroupNorm):
            conv0  = StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            conv0 = nn.Conv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)

        self.root = nn.Sequential(OrderedDict([
            ('conv', conv0),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf, norm_layer = norm_layer))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf, norm_layer = norm_layer)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2, norm_layer = norm_layer))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf, norm_layer = norm_layer)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2, norm_layer = norm_layer))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf, norm_layer = norm_layer)) for i in range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2, norm_layer = norm_layer))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf, norm_layer = norm_layer)) for i in range(2, block_units[3] + 1)],
            ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
            ('gn', norm_layer(2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[...,0,0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
                self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')
    
    


layers_bit_resnet_cfg = OrderedDict([
    ('BiT_R50x1', ([3, 4, 6, 3], 1)),
    ('BiT_R50x3', ([3, 4, 6, 3], 3)),
    ('BiT_R101x1', ([3, 4, 23, 3], 1)),
    ('BiT_R101x3', ([3, 4, 23, 3], 3)),
    ('BiT_R152x2', ([3, 8, 36, 3], 2)),
    ('BiT_R152x4', ([3, 8, 36, 3], 4)),
])

default_bit_resnet_cfg = {
    'in_channels': 3,
    'num_classes': 1000,
    'im_dim': (224,224),
    'layers': layers_bit_resnet_cfg['BiT_R50x1'],
    'resize': True,
    'batchnorm': False,
    'groupnorm': False,
    'layernorm': False,
    'patchify': False,
    'trunc_normal_init': False,
}

from timm.models.layers import LayerNorm2d 
# torch LayerNorm normalizes over the last dimension(s) 
# for the CHW format to get layernorm over channels, timm norm 
class BiT_ResNet(ResNetV2):

    def __init__(self, model_cfg=default_bit_resnet_cfg) -> None:

        num_classes = model_cfg.get('num_classes',default_bit_resnet_cfg['num_classes'])
        layers = model_cfg.get('layers',default_bit_resnet_cfg['layers'])

        if model_cfg['groupnorm']:
            norm_layer = GroupNormPartialWrapper
        elif model_cfg['batchnorm']:
            norm_layer = nn.BatchNorm2d
        elif model_cfg['layernorm']:
            norm_layer = LayerNorm2d
        else:
            norm_layer = None

        super(BiT_ResNet, self).__init__(*layers, head_size=num_classes, zero_head=False, norm_layer = norm_layer)

        if model_cfg.get('patchify', False):
            wf = layers[1]
            if issubclass(norm_layer, nn.GroupNorm):
                conv0  = StdConv2d(3, 64*wf, kernel_size=4, stride=4, bias=False)
            else:
                conv0 = nn.Conv2d(3, 64*wf, kernel_size=4, stride=4, padding=3, bias=False)
            norm0 = norm_layer(64*wf)
            self.root = nn.Sequential(OrderedDict([
                ('conv', conv0),
                ('norm', norm0)
            ]))

        self._initialize_weights(trunc_normal=model_cfg.get('trunc_normal_init', False))

    def _initialize_weights(self, trunc_normal = False):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if not(trunc_normal):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


normalize_transform = Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))


class BitResNet(nn.Module):
    
    def __init__(self, checkpoint_path=None, normalize=True):
        super().__init__()
        model_cfg = copy.deepcopy(default_bit_resnet_cfg)
        if '_gn_' in checkpoint_path:
            model_cfg['groupnorm'] = True
        elif '_bn_' in checkpoint_path:
            model_cfg['batchnorm'] = True
        elif '_ln_' in checkpoint_path:
            model_cfg['layernorm'] = True
        else:
            raise ValueError('checkpoint path should have _bn_, _gn_, or _ln_ in name')
        if 'patchify' in checkpoint_path:
            model_cfg['patchify'] = True
        self._model = BiT_ResNet(model_cfg=model_cfg)
        pretrained_data = torch.load(checkpoint_path)
        self._model.load_state_dict(pretrained_data['model_state_dict'])
        self._normalize = normalize
        
    def forward(self, x):
        if self._normalize:
            x = normalize_transform(x)
        return self._model(x)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val

    def get_layers(self):
        blocks = list(self._model.body)
        layers = ([('stem', self._model.root)] +
                  [('conv-block-' + str(i), b) for i, b in zip(range(len(blocks)), blocks)] +
                  [('head', self._model.head)])
        return layers

    def freeze_bottom_k(self, k):
        layers = [l for _, l in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        for i in range(k):
            utils.set_requires_grad(layers[i], False)

    def new_last_layer(self, num_classes):
        num_features = self._model.head.gn.num_channels
        self._model.head.conv = nn.Conv2d(num_features, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def add_probe(self, probe):
        raise NotImplementedError('Not Implemented yet.')

    def get_last_layer(self):
        return self._model.head.conv

    def set_last_layer(self, coef, intercept):
        raise NotImplementedError('Not Implemented yet.')

    def get_feature_extractor(self):
        raise NotImplementedError('Not Implemented yet.')

    def get_features(self, x):
        raise NotImplementedError('Not Implemented yet.')

