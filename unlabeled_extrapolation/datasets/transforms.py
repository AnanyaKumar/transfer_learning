# Custom transforms.

import torchvision.transforms

class Resize(torchvision.transforms.Resize):

    def __init__(self, size, interpolation='bilinear'):
        if interpolation == 'bilinear':
            interp_arg = torchvision.transforms.InterpolationMode.BILINEAR
        if interpolation == 'bicubic':
            interp_arg = torchvision.transforms.InterpolationMode.BICUBIC
        if interpolation == 'nearest':
            interp_arg = torchvision.transforms.InterpolationMode.NEAREST
        super().__init__(size=size, interpolation=interp_arg)
