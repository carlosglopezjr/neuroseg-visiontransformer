import ml_collections
"""
This code is adapted from the implementation in:

Beckschen, 3D-TransUNet repository
https://github.com/Beckschen/3D-TransUNet

Specifically:
vit_configs.py

Modifications:
- removed everything else besides get_b16_config

Original authors retain all rights under the repository’s license.
"""
def get_b16_config():
    """Returns the ViT-B/16 configuration from paper implementation; may be main config"""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config