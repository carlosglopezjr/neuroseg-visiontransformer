import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import interp_methods
import matplotlib.pyplot as plt
import scipy
import ml_collections
import fvcore.nn.weight_init as weight_init


from torch.utils.data import Dataset, DataLoader
from resize_right import resize
from transunet3d_model import TransUNet
from hungarian3d import HungarianMatcher3D, compute_loss_hungarian
from bratsdataloader import create_brats_dataloaders
from trainer import TransUNetTrainer

"""
Script used to train models
"""
DATA_DIR = "./testbatch_Task01_reorg"
# ── CONFIG ─────────────────────────────────────────────────────
#'/content/drive/MyDrive/BraTS2021/extracted' #update path where extracted cases are
# CHECKPOINT_DIR = '/content/drive/MyDrive/277B_checkpoints'
# RESUME_FROM    = None #'/content/drive/MyDrive/277B_checkpoints/checkpoint_ep39.pt'   # set to checkpoint path to resume /content/drive/MyDrive/277B_checkpoints/checkpoint_ep39.pt

#Training parameters
# d, h, w        = 224, 224, 144    # middle range and not too computationall intensive
# BATCH_SIZE     = 2
# NUM_CLASSES    = 4
# INPUT_CHANNELS = 4
# VIT_DEPTH      = 12

#Choose a model from the following options
# ["CNN", "ENCODER-ONLY","DECODER-ONLY", "Full-TransUnet"]
model = "CNN"

def ModelSelectionDataCuration(model, trainer, modality,INPUT_CHANNELS,d,h,w,BATCH_SIZE,NUM_CLASSES,VIT_DEPTH):
        model_configs = ["CNN", "ENCODER-ONLY", "DECODER-ONLY", "FullTransUNet"]
        if model not in model_configs:
             #Raise an exception with message
             raise ValueError(f"Error, got: {model} but expected one of the following {model_configs}")
        if model == "CNN":
            #── CNN only ----------------------------------------------
            tr_gen, val_gen = create_brats_dataloaders(
                mode='CNN',
                data_dir=DATA_DIR,
                modalities=modality,
                patch_size=(d, h, w),
                batch_size=BATCH_SIZE,
            )
            trainer.network = TransUNet(
                input_channels=INPUT_CHANNELS,
                base_num_features=32,
                num_classes=NUM_CLASSES,
                num_pool=5,
                patch_size=[d, h, w],
                is_max=False,                      # ← CNN only (no Tranformer decoder)
                deep_supervision=True,
                is_max_bottleneck_transformer=False,
            ).to(trainer.device)
            return tr_gen, val_gen

        elif model == "ENCODER-ONLY":
            # Encoder only (CNN-Transformer Encoder, standard Unet Decoder) ------
            tr_gen, val_gen = create_brats_dataloaders(
                mode='ENCODER',
                data_dir=DATA_DIR,
                modalities=modality,
                patch_size=(d, h, w),
                batch_size=BATCH_SIZE,
            )
            trainer.network = TransUNet(
                input_channels=INPUT_CHANNELS,
                base_num_features=32,
                num_classes=NUM_CLASSES,
                num_pool=5,
                patch_size=[d, h, w],
                is_max=False,                          # ← no Transformer decoder
                is_max_bottleneck_transformer=True,    # ← ViT encoder at bottleneck
                deep_supervision=True,                 # use CNN seg outputs
                vit_depth=VIT_DEPTH,
            ).to(trainer.device)
            return tr_gen, val_gen

        elif model == "DECODER-ONLY":
            # Decoder Only (CNN encoder, CNN-Transformer for decoder)
            tr_gen, val_gen = create_brats_dataloaders(
                mode='DECODER',
                data_dir=DATA_DIR,
                modalities=modality,
                patch_size=(d, h, w),
                batch_size=BATCH_SIZE,
            )
            trainer.network = TransUNet(
                input_channels=INPUT_CHANNELS,
                base_num_features=32,
                num_classes=NUM_CLASSES,
                num_pool=5,
                patch_size=[d, h, w],
                is_max=True,                           # decoder runs
                is_max_bottleneck_transformer=False,   # ViT skipped
                is_max_cls=True,
                is_max_ds=True,
                is_max_hungarian=True,
                mw=1.0,
                deep_supervision=False,
                vit_depth = VIT_DEPTH,
            ).to(trainer.device)
            return tr_gen, val_gen

        else: # ENCODER + DECODER (FULL TransUNet)
            tr_gen, val_gen = create_brats_dataloaders(
                mode='FullTransUNet',
                data_dir=DATA_DIR,
                modalities=modality,
                patch_size=(d, h, w),
                batch_size=BATCH_SIZE,
            )

            trainer.network = TransUNet(
                input_channels=INPUT_CHANNELS,
                base_num_features=32,
                num_classes=NUM_CLASSES,
                num_pool=5,
                patch_size=[d, h, w],
                is_max=True,                       # ← Transformer decoder
                is_max_cls=True,
                is_max_ds=True,
                is_max_hungarian=True,
                mw=1.0,
                deep_supervision=False,
                is_max_bottleneck_transformer=True,
                vit_depth=12,
            ).to(trainer.device)
            return tr_gen, val_gen


# ── build trainer ───────────────────────────────────────────────
if __name__ == '__main__':
    trainer = TransUNetTrainer(
        output_folder=CHECKPOINT_DIR,
        initial_lr=1e-4,
        max_num_epochs=5,
        num_batches_per_epoch=10,
        num_val_batches_per_epoch=5,
        fp16=True,
        save_every=5,
        num_classes=NUM_CLASSES,
        patch_size=(d, h, w),
        batch_size=BATCH_SIZE,
        plot_title= model
    )


    trainer.initialize()

    tr_gen, val_gen = ModelSelectionDataCuration("CNN",trainer, modality= ('t1', 't1ce',))

    #Create Optimizer and run training
    trainer.optimizer = torch.optim.AdamW(
        trainer.network.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)

    if RESUME_FROM:
        trainer.load_checkpoint(RESUME_FROM)

    trainer.run_training(tr_gen,val_gen)

