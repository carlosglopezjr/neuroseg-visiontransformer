import copy
import math

import vit_configs as configs

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from utils import np2th, activation_fns, LayerScale


"""
This code is adapted from the implementation in:

Beckschen, 3D-TransUNet repository
https://github.com/Beckschen/3D-TransUNet

Specifically:
vit_modeling.py

Modifications:
- modified line 99 in Beckschen to 'activation_fns["gelu"]' within MLP class, imported activation_fns from utils.py
- changed Block class to VITBlock class
- modified line 185 and 190 in Beckschen to 'x = h + self.ls1(x)' and 'x = h + self.ls2(x)' respectively - this adds residuals to newly scaled info
- removed 'def load_from' from within Beckschen Block class (our VITBlock class)
- removed 'def load_from' from within Beckschen Transformer class


Original authors retain all rights under the repository’s license.
"""
class Attention(nn.Module):
    """
    class defining SELF attention blocks
    """
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

    
class MLP(nn.Module):
    """
    Multi-Layer Perceptron block 
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = activation_fns["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings 
    (gives position embeddings spatial identity)
    """
    def __init__(self, config, feat_size, in_channels = 3):
        super(Embeddings, self).__init__()
        self.config = config

        #patch_size = _pair(config.patches["size"])
        n_patches = feat_size[0] * feat_size[1] * feat_size[2]
        patch_size = 1

        self.hybrid = False # not a hybrid model

        self.patch_embeddings = nn.Conv3d(in_channels = in_channels,
                                        out_channels = config.hidden_size,
                                        kernel_size = patch_size,
                                        stride = patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None              # not a hybrid model
        x = self.patch_embeddings(x) # (B, hidden.n_patches^(1/2), n_patches(1/2))
        x = x.flatten(2)             # collapse into a sequence
        x = x.transpose(-1, -2)      # (B, n_patches, hidden), swaps channels and sequence so that tokens are rows

        embeddings = x + self.position_embeddings #position_embeddings initally starts at zero, added to x to give spatial identity
        embeddings = self.dropout(embeddings)
        return embeddings, features 
    
class VITBlock(nn.Module):
    """
    Implements equations 2 and 3 from paper,
    feeds residuals to attention ouput, then uses that on MLP layer to calculate l-th layer
    """
    def __init__(self, config, vis, use_layer_scale):
        super(VITBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config, vis)

        #Layerscale multiplies vector with small intialized values to supresse new informaton and stabilize models
        self.ls1 = LayerScale(config.hidden_size, init_values=1e-5) if use_layer_scale else nn.Identity()
        self.ls2 = LayerScale(config.hidden_size, init_values=1e-5) if use_layer_scale else nn.Identity()

    def forward(self, x):
        h = x                          #save h as residual
        x = self.attention_norm(x)     #runs layer norm
        x, weights = self.attn(x)      # x is attention output, self.attn(x) computes 12-head self attention
        x = h + self.ls1(x)            #should be adding residual to newly scaled info

        # adding previous x to MLP branch below
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + self.ls2(x)
        return x, weights
    

class Encoder(nn.Module):
    """
    Chains 12 ViTBlock instances via copy.deepcopy,
    vis = False by default to keep attn_weights empty to save memory (not important to training)
    """
    def __init__(self, config, vis, use_layer_scale):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = VITBlock(config, vis,use_layer_scale)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    

class Transformer(nn.Module):
    """
    High level wrapper, takes CNN bottleneck -> embeds -> encodes and
    reshapes into expected size for CNN decoder 
    """
    def __init__(self, config, feat_size, vis, feat_channels, use_layer_scale):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, feat_size=feat_size, in_channels=feat_channels)
        self.encoder = Encoder(config, vis, use_layer_scale)

    def forward(self, input_ids):

        embedding_output, features = self.embeddings(input_ids) #receives CNN bottleneck
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)

        B, n_patch, hidden = encoded.size()                     # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w, d = input_ids.shape[2:]
        x = encoded.permute(0, 2, 1)
        encoded = x.contiguous().view(B, hidden, h, w, d)
        return encoded, attn_weights


CONFIGS = {'ViT-B_16': configs.get_b16_config()}