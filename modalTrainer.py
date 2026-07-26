import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import interp_methods
import matplotlib.pyplot as plt
import scipy
import ml_collections
import torch
import fvcore.nn.weight_init as weight_init
import os

from torch.utils.data import Dataset, DataLoader
from resize_right import resize
from transunet3d_model import TransUNet
from hungarian3d import HungarianMatcher3D, compute_loss_hungarian
from bratsdataloader import create_brats_dataloaders
from trainer import TransUNetTrainer, maybe_to_torch
from run_trainer import ModelSelectionDataCuration
from bratsTestBatchloader import create_brats_test_dataloader


import logging
import sys
import traceback
from pathlib import Path

LOG_DIR = Path("/vol/logs")   # assuming /vol is your mounted Modal Volume
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting training...")



    TRAIN_DIR = "/mnt/brats-data/BraTS2021/extracted"
    TEST_DIR  = "/mnt/brats-data/BraTS2021/testbatch"

    print(os.path.exists(TRAIN_DIR), TRAIN_DIR)
    print(os.path.exists(TEST_DIR), TEST_DIR)
    print(os.listdir("/mnt/brats-data/BraTS2021")[:10])

    VOLUME_ROOT = "/mnt/brats-data"

    DATA_DIR = f"{VOLUME_ROOT}/BraTS2021/extracted"
    TEST_DIR = f"{VOLUME_ROOT}/BraTS2021/testbatch"

    CHECKPOINT_DIR = f"{VOLUME_ROOT}/outputs/checkpoints"
    GRAPH_DIR = f"{VOLUME_ROOT}/outputs/graphs"
    MODEL_DIR = f"{VOLUME_ROOT}/outputs/models"
    PREDS_DIR = f"{VOLUME_ROOT}/outputs/predictions"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    # ── CONFIG ─────────────────────────────────────────────────────
    # DATA_DIR       = "/mnt/brats-data/BraTS2021/extracted" #update path where extracted cases are
    # CHECKPOINT_DIR = '/content/drive/MyDrive/277B_checkpoints'
    RESUME_FROM    = None #'/content/drive/MyDrive/277B_checkpoints/checkpoint_ep39.pt'   # set to checkpoint path to resume /content/drive/MyDrive/277B_checkpoints/checkpoint_ep39.pt

    #for when H100 is available
    # d, h, w        = 160, 224, 192
    # BATCH_SIZE     = 2
    # NUM_CLASSES    = 4      # BraTS: background + 3 tumor regions
    # INPUT_CHANNELS = 4      # 4 MRI modalities
    # VIT_DEPTH      = 12

    d, h, w        = 144, 224, 224   # middle range and not too computationall intensive
    BATCH_SIZE     = 2
    NUM_CLASSES    = 4
    VIT_DEPTH      = 12
    MODALITIY      = ('t1', 't1ce',) #('t1', 't1ce', 't2', 'flair')
    INPUT_CHANNELS = len(MODALITIY)


    #Choose a model from the following options
    # ["CNN", "ENCODER-ONLY","DECODER-ONLY", "FullTransUNet"]
    model = "FullTransUNet"

    # ── build trainer ───────────────────────────────────────────────
    trainer = TransUNetTrainer(
        output_folder=CHECKPOINT_DIR,
        initial_lr=1e-4,
        max_num_epochs=100,
        num_batches_per_epoch=100,
        num_val_batches_per_epoch=50,
        fp16=True,
        save_every=5,
        num_classes=NUM_CLASSES,
        patch_size=(d, h, w),
        batch_size=BATCH_SIZE,
        plot_title= model
    )


    trainer.initialize()



    if model == "CNN":
        #── CNN only ----------------------------------------------
        tr_gen, val_gen = create_brats_dataloaders(
            mode='CNN',
            data_dir=DATA_DIR,
            modalities = MODALITIY,
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

    elif model == "ENCODER-ONLY":
    # Encoder only (CNN-Transformer Encoder, standard Unet Decoder) ------
        tr_gen, val_gen = create_brats_dataloaders(
            mode='ENCODER',
            data_dir=DATA_DIR,
            modalities = MODALITIY,
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

    elif model == "DECODER-ONLY":
    # Decoder Only (CNN encoder, CNN-Transformer for decoder)
        tr_gen, val_gen = create_brats_dataloaders(
            mode='DECODER',
            data_dir=DATA_DIR,
            modalities = MODALITIY,
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

    else: # ENCODER + DECODER (FULL TransUNet)
        tr_gen, val_gen = create_brats_dataloaders(
            mode='FullTransUNet',
            data_dir=DATA_DIR,
            modalities = MODALITIY,
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

    #Create Optimizer and run training
    trainer.optimizer = torch.optim.AdamW(
        trainer.network.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)

    if RESUME_FROM:
        trainer.load_checkpoint(RESUME_FROM)

    trainer.run_training(tr_gen, val_gen)

    result = full_results    = trainer.evaluate_dice(val_gen, regions=('ET','TC','WT'))
    #et_only_results = trainer.evaluate_dice(val_gen, regions=('ET',))
    #wt_only_results = trainer.evaluate_dice(val_gen, regions=('WT',))
    #tc_only_results = trainer.evaluate_dice(val_gen, regions=('TC',))

    trainer.save_dice_results(
        result,
        model,
        MODALITIY,
        MODEL_DIR)


    # import torch
    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from trainer import TransUNetTrainer, maybe_to_torch
    # from run_trainer import ModelSelectionDataCuration
    # from bratsTestBatchloader import create_brats_test_dataloader


    os.makedirs(PREDS_DIR, exist_ok=True)

    DATA_DIR = TEST_DIR
    CHECKPOINT_DIR = MODEL_DIR
    #Ask user which model they would like to run ----------------------------------------------------------------
    #choose model configuration and ensure correct brats loader

    #ask user for model
    model_configs = ["CNN", "ENCODER-ONLY", "DECODER-ONLY", "FullTransUNet"]
    #model = input(f"Please select model you would like to run, choose one of the following \n{model_configs} ")
    model = "FullTransUNet"
    if model not in model_configs:
        raise ValueError(f"Invalid model: {model}. Choose from {model_configs}")

    model_to_mode = {'CNN':'CNN','ENCODER-ONLY':'ENCODER','DECODER-ONLY':'DECODER','FullTransUNet':'FullTransUNet'}

    #ask user which modality so pre-trained model can be chosen (well have single modality and multmodal models to choose from)
    #modality = input(f"Which pretrained model?") #later this will be Single Modality or Multimodality --> which modality/subregion for now just this to work
    modality = ('t1', 't1ce',)

    #for CNN-> /mnt/brats-data/outputs/models/CNN_SingMod_TC_T1ceT1_model_final.pt
    #-> original /content/277B_FinalProject/Trans-U-Net/pretrained_models/CNN_SingMod_TC_T1ceT1_model_final.pt'
    model_to_pretrained_model = {"CNN":'/mnt/brats-data/outputs/models/CNN_SingMod_TC_T1ceT1_model_final.pt',
                                    "FullTransUNet":'/mnt/brats-data/outputs/models/model_final.pt'}
    PRE_TRAINED_MODEL = model_to_pretrained_model[model]

    # parameters
    d, h, w        = 128, 128, 128    # middle range and not too computationall intensive
    BATCH_SIZE     = 1
    NUM_CLASSES    = 4
    MODALITY      = modality #('t1', 't1ce', 't2', 'flair')
    INPUT_CHANNELS = len(modality)
    VIT_DEPTH      = 12

    # ── build trainer ───────────────────────────────────────────────
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


    _ , _ = ModelSelectionDataCuration(model,trainer,modality,INPUT_CHANNELS,d,h,w,BATCH_SIZE,NUM_CLASSES) #only need the model and not the train and val datasets, passed by reference

    #Create Optimizer and run training
    trainer.optimizer = torch.optim.AdamW(
        trainer.network.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)

    #load a trained model from checkpoiunt ----------------------------------------------------------------
    trainer.load_checkpoint(PRE_TRAINED_MODEL)
    trainer.network.eval()
    #trainer.run_training(val_gen)

    #load in a Brats scan ----------------------------------------------------------------
    test_gen = create_brats_test_dataloader(
                    mode= model_to_mode[model],
                    data_dir=DATA_DIR,
                    modalities=modality,
                    patch_size=(d, h, w),
                    batch_size=BATCH_SIZE,
                )
    # preprocess it the same way brats dataloader does ----------------------------------------------------------------

    num_cases = len(test_gen.dataloader.dataset)
    for i in range(num_cases):

        batch = next(test_gen)
        case_name = batch['cases'][0]
        data   = maybe_to_torch(batch['data']).to(trainer.device)

        if 'target' in batch:
            #Transformer decoder mode
            target = [{k: v.to(trainer.device) for k, v in t.items()} for t in batch['target']]

        else:
            #CNN mode
            target = [{
                'labels': l.to(trainer.device),
                'masks' : m.to(trainer.device),
            } for l, m in zip(batch['labels'], batch['masks'])]

    with torch.no_grad():
        output = trainer.network(data)

        if isinstance(output, dict):
            pred_mask = output['pred_masks'].sigmoid().argmax(dim=1) # (B, N_queries, D, H, W) argmax of each query
            pred_logits = output['pred_logits'].argmax(dim=-1)        # (B, N_queries, num_classes+1) class scores

            B = pred_mask.shape[0]
            batch_idx = torch.arange(B).view(B,1,1,1).expand_as(pred_mask) #gets pred_mask shape, reshapes into 4D tensor, uses expand_as to match tensor dimensions of pred_mask

            pred_seg = pred_logits[batch_idx,pred_mask]

            D, H, W =  pred_mask.shape[1:] #gets dimensions from pred_mask

        else:
            #handles only CNN and Encoder output
            seg = output[0] if isinstance(output, tuple) else output
            pred_seg = seg.argmax(dim=1) # (B, C, D, H, W) - > (B, D, H, W)
            B, C, D, H, W = seg.shape

        #Buid integer label map-----------------
        gt_seg = torch.zeros(B, D, H, W,
                            dtype=torch.long,
                            device=trainer.device)
        for b, t in enumerate(target):
            for mask, label in zip(t['masks'], t['labels']):
                gt_seg[b][mask.bool()] = label

        #visualization ----------------------------------------------------------------
        #Get first items in batch
        pred_slice = pred_seg[0]
        gt_slice = gt_seg[0]

        #identify which slices have the 'best' tumor visualization
        tumor_mask = gt_slice.bool()
        tumor_per_slice = tumor_mask.sum(dim=(1,2))
        best_d = tumor_per_slice.argmax()

        pred_2d = pred_seg[0, best_d].cpu().numpy()
        gt_2d = gt_seg[0, best_d].cpu().numpy()
        image_slice = data[0,1,best_d].cpu().numpy()

        # ── visualization ─────────────────────────────────────────────
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # subplot 1 — MRI only
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title(f'T1ce  (slice {best_d})')
        ax1.axis('off')

        # subplot 2 — ground truth overlay
        ax2.imshow(image_slice, cmap='gray')
        ax2.imshow(gt_2d, alpha=0.5, cmap='jet', vmin=0, vmax=3)
        ax2.set_title('ground truth')
        ax2.axis('off')

        # subplot 3 — prediction overlay
        ax3.imshow(image_slice, cmap='gray')
        ax3.imshow(pred_2d, alpha=0.5, cmap='jet', vmin=0, vmax=3)
        ax3.set_title('prediction')
        ax3.axis('off')

        plt.suptitle(f'{case_name}  |  {model}  |  best slice: {best_d}',
                    fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(PREDS_DIR, f'{case_name}_pred_vs_gt.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    logger.info("Training finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training crashed with an exception:")
        raise