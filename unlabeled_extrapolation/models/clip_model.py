# Note: this model does normalization.
import clip
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
import os

from . import model_utils

from torchvision.transforms import Normalize


MODELS = {'RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16',
          'ViT-L/14', 'ViT-L/14@336px'}

normalize_transform = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))


def build_model_scratch(model_name, device):
    model_path = clip.clip._download(clip.clip._MODELS[model_name], os.path.expanduser("~/.cache/clip"))
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = clip.model.CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clip.model.convert_weights(model)
    return model.eval().to(device)


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


class ClipModel(nn.Module):

    def __init__(self, model_name, scratch=False):
        # If scratch is True, then we randomly initialize weights.
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        if scratch:
            model = build_model_scratch(model_name, device=self._device)
        else:
            model, _ = clip.load(model_name, device=self._device)
        self._model_name = model_name
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

    def get_layers(self):
        visual = self._model.visual
        if self._model_name in {'ViT-B/32', 'ViT-B/16',
                                'ViT-L/14', 'ViT-L/14@336px'}:
            layers = [
                ('patch_embed', visual.conv1),
                ('ln_pre', visual.ln_pre),  # To streamline number of layers with CLIP.
                ('pos_embed', model_utils.ParamWrapperModule(visual.positional_embedding)),
                ('cls_token', model_utils.ParamWrapperModule(visual.class_embedding)),
            ]
            blocks = visual.transformer.resblocks
            for i, block in zip(range(len(blocks)), blocks):
                layers += [
                    ('trans' + str(i) + '_norm1', block.ln_1),
                    ('trans' + str(i) + '_attn', block.attn),
                    ('trans' + str(i) + '_norm2', block.ln_2),
                    ('trans' + str(i) + '_mlp', block.mlp),
                ]
            layers += [('post_norm', visual.ln_post)]
            layers += [('head', self.get_last_layer())]
        elif self._model_name in {'RN50'}:
            layers = [
                ('conv1', visual.conv1),
                ('bn1', visual.bn1),
                ('conv2', visual.conv2),
                ('bn2', visual.bn2),
                ('conv3', visual.conv3),
                ('bn3', visual.bn3),
                ('layer1', visual.layer1),
                ('layer2', visual.layer2),
                ('layer3', visual.layer3),
                ('attnpool', visual.attnpool)]
            if self._classifier is not None:
                layers = layers + [('head', self._classifier)]
        else:
            raise NotImplementedError
        return layers

    def get_num_trans_layers(self):
        visual = self._model.visual
        return len(visual.transformer.resblocks)

    def freeze_bottom_trans(self, num_trans_freeze, freeze_embed):
        num_trans_layers = self.get_num_trans_layers()
        if num_trans_freeze == num_trans_layers:
            # If we are tuning the head, don't tune the post layer norm.
            num_layers_freeze = num_trans_freeze * 4 + 5
        elif num_trans_freeze == 0:
            if freeze_embed:
                num_layers_freeze = 2
            else:
                num_layers_freeze = 0
        else:
            num_layers_freeze = num_trans_freeze * 4 + 4
        # Set all grads to True.
        set_requires_grad(self, True)
        # Call freeze.
        # TODO: Q: should we freeze pos embedding and cls token?
        self.freeze_bottom_k(num_layers_freeze)

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
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        for i in range(k):
            set_requires_grad(layers[i], False)

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
