# pre-defined functions and variables
import torch
import torch.nn as nn
from resize_right import resize
import interp_methods
#from . import vit_configs as configs
"""
This code is adapted from the implementation in:

Beckschen, 3D-TransUNet repository
https://github.com/Beckschen/3D-TransUNet

Specifically:
vit_modeling.py and transunet3d_model.py

Modifications:
- This file stiches together the following function and classes in a single file for ease of use
- np2th and Layerscale obtained from vit_modeling.py
- Upsample obtained from transunet3d_model.py

- the ImageDownsample class was a custom class Class made 
  for downsampling 5D tensors from 3D image data to be able to run on local computers.
  
Original authors retain all rights under the repository’s license.
"""

swish =  lambda x: x * torch.sigmoid(x)
activation_fns = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish" : swish}

def np2th(weights, conv = False):
    """Possibly convert HWIO to OIHW"""
    if conv:
        weights = weights.tranpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values = 1e-5, inplace = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Upsample(nn.Module):
    def __init__(self, size = None, scale_factor = None, mode = 'trilinear', align_corners = False):
        super(Upsample, self).__init__()

        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size = self.size, scale_factor = self.scale_factor, mode = self.mode,
        align_corners = self.align_corners)

class ImageDownsample():
    """
    Class made for downsampling 5D tensors from 3D image data to be able to run on local computers.
    Image is assumed to be provided in x, y, z format. 
    """
    def __init__(self, img_data, max_dhw = (64, 64, 64), is_numpy = True):
        if is_numpy:
            w, h, d = img_data.shape
        else:
            w, h, d = img_data.size()
        assert (d, h, w) > max_dhw, "Are you sure you want to downsample? Check the dimensions again!"
        self.scale_factors = (max_dhw[2]/w, max_dhw[1]/h, max_dhw[0]/d) 
    
    def __call__(self, x: torch.Tensor):
        new_x = resize(x, scale_factors=self.scale_factors, out_shape=None,
                    interp_method=interp_methods.lanczos2, support_sz=None,
                    antialiasing=True, by_convs=False, scale_tolerance=None,
                    max_numerator=10, pad_mode='constant')
        return new_x 

#CONFIGS = {'ViT-B_16': configs.get_b16_config()}