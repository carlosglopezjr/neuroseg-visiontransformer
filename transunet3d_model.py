import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

from copy import deepcopy
from torch import nn

from vit_model import Transformer
from vit_model import CONFIGS as CONFIGS_ViT
from utils import Upsample
from hungarian3d import HungarianMatcher3D

softmax_helper = lambda x: F.softmax(x, 1)

class InitWeights(object):
    def __init__(self, neg_slope = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a = self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonLin(nn.Module):
    def __init__(self, input_channels, output_channels,
                conv_op = nn.Conv3d, conv_kwargs = None,
                norm_op = nn.BatchNorm3d, norm_op_kwargs = None,
                dropout_op = nn.Dropout2d, dropout_op_kwargs = None,
                nonlin = nn.LeakyReLU, nonlin_kwargs = None):
        super(ConvDropoutNormNonLin, self).__init__()

        # default arg values
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))
    

class Conv3dLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                conv_op = nn.Conv3d, conv_kwargs = None,
                norm_op = nn.BatchNorm3d, norm_op_kwargs = None,
                dropout_op = nn.Dropout3d, dropout_op_kwargs = None,
                nonlin = nn.LeakyReLU, nonlin_kwargs = None,
                first_stride = None, basic_block = ConvDropoutNormNonLin):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(Conv3dLayers, self).__init__()
        # change below to ModuleList?
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))
        
    def forward(self, x):
        return self.blocks(x)
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device): # to set to gpu for example
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError("This method for this class has not been implemented!")


class TransUNet(NeuralNetwork):
    MAX_NUM_FILTERS_3D = 320
    MAX_FILTERS_2D = 480

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False, # TODO default False
                 max_num_features=None, basic_block=ConvDropoutNormNonLin,
                 seg_output_use_bias=False,
                 patch_size=None, is_vit_pretrain=False, 
                 vit_depth=12, vit_hidden_size=768, vit_mlp_dim=3072, vit_num_heads=12,
                 max_msda='', is_max_ms=True, is_max_ms_fpn=False,is_masking=False, is_masking_argmax=False, max_n_fpn=4, max_ms_idxs=[-4,-3,-2], max_ss_idx=0,
                 is_max=False,is_masked_attn=False, is_max_ds=False,is_max_bottleneck_transformer=False, max_seg_weight=1.0, max_hidden_dim=256, max_dec_layers=10,
                 mw = 0.5,
                 is_fam = False,is_max_hungarian=False, num_queries=None, is_max_cls=False,
                 point_rend=False, num_point_rend=None, no_object_weight=None, is_mhsa_float32=False, no_max_hw_pe=False,
                 max_infer=None, cost_weight=[2.0, 5.0, 5.0], vit_layer_scale=False, decoder_layer_scale=False):
        super(TransUNet, self).__init__()
        self.is_fam = is_fam

        self.is_max, self.max_msda, self.is_max_ms, self.is_max_ms_fpn, self.max_n_fpn, self.max_ss_idx, self.mw = is_max, max_msda, is_max_ms, is_max_ms_fpn, max_n_fpn, max_ss_idx, mw
        self.is_max_cls = is_max_cls
        self.max_ms_idxs = max_ms_idxs

        self.is_masked_attn, self.is_max_ds = is_masked_attn, is_max_ds

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.is_max_bottleneck_transformer = is_max_bottleneck_transformer

        if conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d # maybe want adaptive or average pooling
            transpconv = nn.ConvTranspose3d

            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [[1,2,2]] + [[2, 2, 2]] * (num_pool-1)
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [[1,3,3]] + [[3, 3, 3]] * (num_pool)
        else:
            raise ValueError("Unsupported convolution dimensionality")
        

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl]) 

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []

        self.fams = [] #for feature alignment module


        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool): #num_pool set to 5
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            #print("Using Convolutional Encoder....")
            self.conv_blocks_context.append(Conv3dLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
   
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        
        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            Conv3dLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            Conv3dLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                Conv3dLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                Conv3dLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        if self.is_fam:
            self.fams = nn.ModuleList(self.fams)

        if self.do_ds:
            self.seg_outputs = []
            for ds in range(len(self.conv_blocks_localization)):
                self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                                1, 1, 0, 1, 1, seg_output_use_bias))
            self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.upscale_logits_ops = []
        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        #Transformer configuration
        if self.is_max_bottleneck_transformer:
            #print("Using Transformer Encoder..")
            self.patch_size = patch_size #e.g [d, h, w]
            config_vit = CONFIGS_ViT['ViT-B_16']
            config_vit.transformer.num_layers = vit_depth
            config_vit.hidden_size = vit_hidden_size # 768
            config_vit.transformer.mlp_dim = vit_mlp_dim # 3072
            config_vit.transformer.num_heads = vit_num_heads # 12
            self.conv_more = nn.Conv3d(config_vit.hidden_size, output_features, 1)
            num_pool_per_axis = np.prod(np.array(pool_op_kernel_sizes), axis=0)
            num_pool_per_axis = np.log2(num_pool_per_axis).astype(np.uint8)
            feat_size = [int(self.patch_size[0]/2**num_pool_per_axis[0]), int(self.patch_size[1]/2**num_pool_per_axis[1]), int(self.patch_size[2]/2**num_pool_per_axis[2])]
            self.transformer = Transformer(config_vit, feat_size=feat_size, vis=False, feat_channels=output_features, use_layer_scale=vit_layer_scale)
        #     #if is_vit_pretrain:
        #         #self.transformer.load_from(weights=np.load(config_vit.pretrained_path))

        # Max PPB+ configuration (i.e MultisScaleStandardTransformationDecoder)
        if self.is_max:
            cfg = {
                    "num_classes": num_classes,
                    "hidden_dim": max_hidden_dim,
                    "num_queries": num_classes if num_queries is None else num_queries, # N=K if 'fixed matching', else default=100,
                    "nheads": 8,
                    "dim_feedforward": max_hidden_dim * 8, # 2048,
                    "dec_layers": max_dec_layers, # 9 decoder layers, add one for the loss on learnable query?
                    "pre_norm": False,
                    "enforce_input_project": False,
                    "mask_dim": max_hidden_dim, # input feat of segm head?
                    "non_object": False,
                    "use_layer_scale": decoder_layer_scale,
            }
            cfg['non_object'] = is_max_cls

            input_proj_list = [] # from low to high resolution
            decoder_channels = [320, 256, 128, 64, 32, 32]
            # use multi-scale feature as Transformer decoder input in the future?
            if self.is_max_ms: # use multi-scale feature as Transformer decoder input
                if self.is_max_ms_fpn:
                    #print("...We are in Branch 1")
                    for idx, in_channels in enumerate(decoder_channels[:max_n_fpn]): # max_n_fpn=4: 1/32, 1/16, 1/8, 1/4
                        input_proj_list.append(nn.Sequential(
                            nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, max_hidden_dim),
                            nn.Upsample(size=(int(patch_size[0]/2), int(patch_size[1]/4), int(patch_size[2]/4)), mode='trilinear')
                        )) # proj to scale (1, 1/2, 1/2), TODO: init
                    self.input_proj = nn.ModuleList(input_proj_list)
                    self.linear_encoder_feature = nn.Conv3d(max_hidden_dim * max_n_fpn, max_hidden_dim, 1, 1) # concat four-level feature
                else:
                    for idx, in_channels in enumerate([decoder_channels[i] for i in self.max_ms_idxs]):
                        #print("...WE ARE IN BRANCH 2\n")
                        input_proj_list.append(nn.Sequential(
                            nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, max_hidden_dim),
                        ))
                    self.input_proj = nn.ModuleList(input_proj_list)

                # self.linear_mask_features =nn.Conv3d(decoder_channels[max_n_fpn-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,) # low-level feat, dot product Trans-feat
                self.linear_mask_features =nn.Conv3d(decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,) # following SingleScale, high-level feat, obtain seg_map
            else:
                self.linear_encoder_feature = nn.Conv3d(decoder_channels[max_ss_idx], cfg["mask_dim"], kernel_size=1)
                self.linear_mask_features = nn.Conv3d(decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,)
            
            if self.is_masked_attn:
                from mask2former_transformer_decoder3d import MultiScaleMaskedTransformerDecoder3d
                cfg['num_feature_levels'] = 1 if not self.is_max_ms or self.is_max_ms_fpn else 3
                cfg["is_masking"] = True if is_masking else False
                cfg["is_masking_argmax"] = True if is_masking_argmax else False
                cfg["is_mhsa_float32"] = True if is_mhsa_float32 else False
                cfg["no_max_hw_pe"] = True if no_max_hw_pe else False
                self.predictor = MultiScaleMaskedTransformerDecoder3d(in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg)
            else:
                from mask2former_transformer_decoder3d import StandardTransformerDecoder
                cfg["dropout"], cfg["enc_layers"], cfg["deep_supervision"] = 0.1, 0, False
                self.predictor = StandardTransformerDecoder(in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg)


    def forward(self, x):
        skips = []
        seg_outputs = []

        # --- Encoder ---
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x) #conv block
    
            skips.append(x)                   #save for skip connection 
            if not self.convolutional_pooling:
                x = self.td[d](x)
        
        x = self.conv_blocks_context[-1](x)    # bottleneck conv

        ######### TransUNet (Transformer Encoder) #########
        if self.is_max_bottleneck_transformer:
            x, attn = self.transformer(x) # [b, hidden, d/8, h/16, w/16] #global self-attention
            x = self.conv_more(x)         #projects back to right channels

        ds_feats = [] # obtain multi-scale feature
        ds_feats.append(x)


        for u in range(len(self.tu)):
            #This bottom block is only used with the feature alignemnet module is active...
            if  u<len(self.tu)-1 and isinstance(self.is_fam, str) and self.is_fam.startswith('fam_down'):
                skip_down = Upsample(size=x.shape[2:])(skips[-(u + 1)]) if x.shape[2:]!=skips[-(u + 1)].shape[2:] else skips[-(u + 1)]
                x_align = self.fams[u](x, x_l=skip_down)
                print("ENTERED FIRST IF STATEMENT")
                x = x + x_align

            x = self.tu[u](x) # merely an upsampling or transposeconv operation

            if isinstance(self.is_fam, bool) and self.is_fam:
                x_align = self.fams[u](x, x_l=skips[-(u + 1)])
                print("ENTERED SECOND IF STATEMENT")
                x = x + x_align

            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if self.do_ds:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            ds_feats.append(x)
        

        #for printing ds_feats (downscale features) architecture visualization
        # print(f"ds_feats has {len(ds_feats)} entries:")
        # for i, f in enumerate(ds_feats):
        #     print(f"  ds_feats[{i}] (neg idx {i-len(ds_feats)}): {f.shape}")

        ######### Max PPB+ #########
        if self.is_max:
            if self.is_max_ms: # is_max_ms_fpn
                multi_scale_features = []
                ms_pixel_feats = ds_feats[:self.max_n_fpn] if self.is_max_ms_fpn else [ds_feats[i] for i in self.max_ms_idxs]
                
                for idx, f in enumerate(ms_pixel_feats): 

                    f = self.input_proj[idx](f) # proj into same spatial/channel dim , but transformer_decoder also project to same mask_dim 
                    multi_scale_features.append(f)
                transformer_decoder_in_feature = self.linear_encoder_feature(torch.cat(multi_scale_features, dim=1)) if self.is_max_ms_fpn else multi_scale_features  # feature pyramid
                mask_features = self.linear_mask_features(ds_feats[-1]) # following SingleScale
            else:
                transformer_decoder_in_feature = self.linear_encoder_feature(ds_feats[self.max_ss_idx])
                mask_features = self.linear_mask_features(ds_feats[-1])
            
            predictions = self.predictor(transformer_decoder_in_feature, mask_features, mask=None)

            if self.is_max_cls and self.is_max_ds:
                if self._deep_supervision and self.do_ds:
                    print("We are within the 2nd if statement...")
                    return [predictions] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
                return predictions

            elif self.is_max_ds and not self.is_max_ms and self.mw==1.0: # aux output of max decoder
                aux_out = [p['pred_masks'] for p in predictions['aux_outputs']] # ascending order
                all_out =  [predictions["pred_masks"]] + aux_out[::-1] # reverse order, w/o sigmoid activation
                return tuple(all_out)
            elif not self.is_max_ds and self.mw==1.0:
                print("WE FAILED")
                raise NotImplementedError
            else:
                raise NotImplementedError
        #############################

        #FOR WHEN USING CNN-ONLY SETTINGS
        if self._deep_supervision and self.do_ds: # assuming turn off ds
            a = tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])

            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:

            return seg_outputs[-1]
