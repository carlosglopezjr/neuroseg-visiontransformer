import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trainer import TransUNetTrainer, maybe_to_torch
from run_trainer import ModelSelectionDataCuration
from bratsTestBatchloader import create_brats_test_dataloader


MM_MODEL_TO_CHECKPOINT = {
    "CNN": "./ModelOutputs/AllModels/CNN_MM.pt",
    "ENCODER-ONLY": "pretrained_models/ENC_MM.pt",
    "DECODER-ONLY": "pretrained_models/DEC_MM.pt",
    "FullTransUNet": "pretrained_models/FULL_MM.pt",
}

SM_MODEL_TO_CHECKPOINT = {
    "CNN": {
        "t1": "pretrained_models/CNN_SM_T1.pt",
        "t1ce": "pretrained_models/CNN_SM_T1CE.pt",
        "t1t1ce": "pretrained_models/CNN_SM_T1T1CE.pt",
        "flair": "pretrained_models/CNN_SM_F.pt",
    },
    "ENCODER-ONLY": {
        "t1": "pretrained_models/ENC_SM_T1.pt",
        "t1ce": "pretrained_models/ENC_SM_T1CE.pt",
        "t1t1ce": "pretrained_models/ENC_SM_T1T1CE.pt",
        "flair": "pretrained_models/ENC_SM_F.pt",
    },
    "DECODER-ONLY": {
        "t1": "pretrained_models/DEC_SM_T1.pt",
        "t1ce": "pretrained_models/DEC_SM_T1CE.pt",
        "t1t1ce": "pretrained_models/DEC_SM_T1T1CE.pt",
        "flair": "pretrained_models/DEC_SM_F.pt",
    },
    "FullTransUNet": {
        "t1": "pretrained_models/FULL_SM_T1.pt",
        "t1ce": "pretrained_models/FULL_SM_T1CE.pt",
        "t1t1ce": "pretrained_models/FULL_SM_T1T1CE.pt",
        "flair": "pretrained_models/FULL_SM_F.pt",
    },
}

def resolve_modality_and_checkpoint(args):
    """
    Determines:
    1. which modalities to use
    2. which checkpoint to load

    Rules:
    - MM: use all four modalities by default: t1, t1ce, t2, flair.
    - SM: allow only t1, t1ce, flair, or t1t1ce.
    - t2 is allowed for MM but not SM.
    - t1t1ce is treated as one checkpoint key, but expanded to ("t1", "t1ce")
      for the dataloader.
    - If --checkpoint is provided, use it instead of dictionary lookup.
    """

    model = args.model
    input_type = args.input_type

    if input_type == "MM":
        if args.modality is None:
            modality = ("t1", "t1ce", "t2", "flair")
        else:
            modality = tuple(args.modality)

        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = MM_MODEL_TO_CHECKPOINT[model]

    elif input_type == "SM":
        allowed_sm_modalities = {"t1", "t1ce", "flair", "t1t1ce"}

        if args.modality is None:
            raise ValueError(
                "For SM/reduced-modality mode, you must provide one modality option. "
                "Example: --input_type SM --modality t1ce"
            )

        if len(args.modality) != 1:
            raise ValueError(
                "For SM/reduced-modality mode, provide exactly one modality option. "
                "Example: --input_type SM --modality t1ce or --input_type SM --modality t1t1ce"
            )

        selected_modality = args.modality[0]

        if selected_modality not in allowed_sm_modalities:
            raise ValueError(
                f"Invalid SM modality: {selected_modality}. "
                f"For SM, choose from: {sorted(allowed_sm_modalities)}. "
                "Note: t2 is only available for MM."
            )

        if selected_modality == "t1t1ce":
            modality = ("t1", "t1ce")
        else:
            modality = (selected_modality,)

        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = SM_MODEL_TO_CHECKPOINT[model][selected_modality]

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return modality, checkpoint_path

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on BraTS test cases using a selected TransUNet configuration."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["CNN", "ENCODER-ONLY", "DECODER-ONLY", "FullTransUNet"],
        help="Model configuration to evaluate."
    )
    parser.add_argument(
    "--input_type",
    type=str,
    required=True,
    choices=["MM", "SM"],
    help="Input type: MM for multimodal, SM for single modality."
    )

    parser.add_argument(
        "--modality",
        nargs="+",
        default=None,
        choices=["t1", "t1ce", "t2", "flair","t1t1ce"],
        help=(
            "Modalities to use. "
            "For MM, defaults to: t1 t1ce t2 flair. "
            "For SM, provide exactly one modality, e.g. --modality t1ce."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./testbatch_Task01_reorg",
        help="Path to the Task01 test data directory."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained model checkpoint, usually model_final.pt."
    )

    parser.add_argument(
        "--preds_dir",
        type=str,
        default="./predictions",
        help="Directory where prediction figures will be saved."
    )

    parser.add_argument(
        "--patch_size",
        nargs=3,
        type=int,
        default=[128, 128, 128],
        metavar=("D", "H", "W"),
        help="Patch size as three integers: D H W. Default: 128 128 128"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference. Default: 1"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of segmentation classes including background. Default: 4"
    )

    parser.add_argument(
        "--vit_depth",
        type=int,
        default=12,
        help="ViT depth used by the model selection function. Default: 12"
    )

    parser.add_argument(
        "--initial_lr",
        type=float,
        default=3e-5,
        help="Initial learning rate used when initializing trainer. Default: 3e-5"
    )

    parser.add_argument(
        "--max_num_epochs",
        type=int,
        default=5,
        help="Trainer max epochs placeholder. Not used for inference, but needed by trainer. Default: 5"
    )

    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=10,
        help="Trainer placeholder value. Default: 10"
    )

    parser.add_argument(
        "--num_val_batches_per_epoch",
        type=int,
        default=5,
        help="Trainer placeholder value. Default: 5"
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Trainer save frequency placeholder. Default: 5"
    )

    parser.add_argument(
        "--no_show",
        action="store_true",
        help="If included, figures will be saved but not displayed with plt.show()."
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Prediction helper
# -------------------------------------------------------------------------

def get_pred_seg(output, num_classes, device):
    """
    Converts model output into a segmentation map with shape:
        (B, D, H, W)

    Handles both:
    1. dictionary output from transformer/query-style models
    2. tensor or tuple output from CNN-style segmentation models
    """

    if isinstance(output, dict):
        pred_masks = output["pred_masks"].sigmoid()      # (B, N_q, D, H, W)
        pred_logits = output["pred_logits"]              # (B, N_q, num_classes + 1)

        B = pred_masks.shape[0]

        pred_seg = torch.zeros(
            B,
            *pred_masks.shape[2:],
            dtype=torch.long,
            device=device
        )

        for b in range(B):
            logits = pred_logits[b]
            masks = pred_masks[b]

            classes = logits.argmax(dim=-1)
            query_per_voxel = masks.argmax(dim=0)
            voxel_classes = classes[query_per_voxel]

            # Treat no-object class as background
            voxel_classes[voxel_classes >= num_classes] = 0

            pred_seg[b] = voxel_classes

    else:
        seg = output[0] if isinstance(output, tuple) else output
        pred_seg = seg.argmax(dim=1)

    return pred_seg


# -------------------------------------------------------------------------
# Ground-truth helper
# -------------------------------------------------------------------------

def build_gt_seg(batch, target, batch_size, depth, height, width, device):
    """
    Reconstructs a dense ground-truth segmentation map from masks and labels.

    Returns:
        gt_seg: (B, D, H, W)
    """

    gt_seg = torch.zeros(
        batch_size,
        depth,
        height,
        width,
        dtype=torch.long,
        device=device
    )

    for b, t in enumerate(target):
        for mask, label in zip(t["masks"], t["labels"]):
            gt_seg[b][mask.bool()] = label

    return gt_seg


# -------------------------------------------------------------------------
# Plotting helper
# -------------------------------------------------------------------------

def save_prediction_figure(
    image_2d,
    gt_2d,
    pred_2d,
    case_name,
    model_name,
    rank,
    slice_idx,
    preds_dir,
    label_legend,
    show=True
):
    """
    Saves a 3-panel visualization:
    1. MRI slice
    2. Ground truth overlay
    3. Prediction overlay
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(bottom=0.15)

    ax1.imshow(image_2d, cmap="gray")
    ax1.set_title(f"T1ce  (slice {slice_idx})")
    ax1.axis("off")

    ax2.imshow(image_2d, cmap="gray")
    ax2.imshow(gt_2d, alpha=0.5, cmap="jet", vmin=0, vmax=3)
    ax2.set_title("ground truth")
    ax2.axis("off")

    ax3.imshow(image_2d, cmap="gray")
    ax3.imshow(pred_2d, alpha=0.5, cmap="jet", vmin=0, vmax=3)
    ax3.set_title("prediction")
    ax3.axis("off")

    fig.legend(
        handles=label_legend,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )

    fig.suptitle(
        f"{case_name}  |  {model_name}  |  rank {rank + 1} slice (d={slice_idx})",
        fontsize=11,
    )

    save_path = os.path.join(
        preds_dir,
        f"{case_name}_rank{rank + 1}_d{slice_idx}.png"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return save_path


# -------------------------------------------------------------------------
# Main inference script
# -------------------------------------------------------------------------

def main():

    # ---------------------------------------------------------------------
    # Basic setup
    # ---------------------------------------------------------------------
    args = parse_args()

    model = args.model
    input_type = args.input_type
    data_dir = args.data_dir
    preds_dir = args.preds_dir

    modality, checkpoint_path = resolve_modality_and_checkpoint(args)

    d, h, w = args.patch_size

    batch_size = args.batch_size
    num_classes = args.num_classes
    input_channels = len(modality)
    vit_depth = args.vit_depth

    os.makedirs(preds_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("BraTS Inference Configuration")
    print("=" * 80)
    print(f"Model:        {model}")
    print(f"Data dir:     {data_dir}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Preds dir:    {preds_dir}")
    print(f"Modalities:   {modality}")
    print(f"Patch size:   {(d, h, w)}")
    print(f"Batch size:   {batch_size}")
    print(f"Num classes:  {num_classes}")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------------------
    # Model mode mapping
    # ---------------------------------------------------------------------

    model_to_mode = {
        "CNN": "CNN",
        "ENCODER-ONLY": "ENCODER",
        "DECODER-ONLY": "DECODER",
        "FullTransUNet": "FullTransUNet",
    }

    mode = model_to_mode[model]

    # ---------------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------------

    label_legend = [
        mpatches.Patch(color="#00007F", label="background (0)"),
        mpatches.Patch(color="#00FFFF", label="edema (1)"),
        mpatches.Patch(color="#FFFF00", label="non-enhancing (2)"),
        mpatches.Patch(color="#FF0000", label="enhancing (3)"),
    ]

    # ---------------------------------------------------------------------
    # Build trainer
    # ---------------------------------------------------------------------

    trainer = TransUNetTrainer(
        output_folder=preds_dir,
        initial_lr=args.initial_lr,
        max_num_epochs=args.max_num_epochs,
        num_batches_per_epoch=args.num_batches_per_epoch,
        num_val_batches_per_epoch=args.num_val_batches_per_epoch,
        fp16=True,
        save_every=args.save_every,
        num_classes=num_classes,
        patch_size=(d, h, w),
        batch_size=batch_size,
        plot_title=model,
    )

    trainer.initialize()

    # ---------------------------------------------------------------------
    # Select model configuration
    # ---------------------------------------------------------------------

    _train_gen, _val_gen = ModelSelectionDataCuration(
        model,
        trainer,
        modality,
        input_channels,
        d,
        h,
        w,
        batch_size,
        num_classes,
        vit_depth,
    )

 
    trainer.optimizer = torch.optim.AdamW(
        trainer.network.parameters(),
        lr=trainer.initial_lr,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------------

    print(f"Loading checkpoint from:\n{checkpoint_path}\n")
    trainer.load_checkpoint(checkpoint_path)

    trainer.network.eval()

    # ---------------------------------------------------------------------
    # Diagnostic batch
    # ---------------------------------------------------------------------

    print("Running diagnostic batch...")

    test_gen = create_brats_test_dataloader(
        mode=mode,
        data_dir=data_dir,
        modalities=modality,
        patch_size=(d, h, w),
        batch_size=batch_size,
    )

    diag_batch = next(test_gen)
    diag_data = maybe_to_torch(diag_batch["data"]).to(trainer.device)

    with torch.no_grad():
        diag_output = trainer.network(diag_data)

    diag_pred = get_pred_seg(diag_output, num_classes, trainer.device)

    print(f"Pred unique labels: {diag_pred.unique()}")
    print("Pred label counts:")

    for label in range(num_classes):
        count = (diag_pred == label).sum().item()
        pct = count / diag_pred.numel() * 100
        print(f"  label {label}: {count:>8,} voxels ({pct:.1f}%)")

    # ---------------------------------------------------------------------
    # Recreate test generator for full inference
    # ---------------------------------------------------------------------

    test_gen = create_brats_test_dataloader(
        mode=mode,
        data_dir=data_dir,
        modalities=modality,
        patch_size=(d, h, w),
        batch_size=batch_size,
    )

    num_cases = len(test_gen.dataloader.dataset)

    print(f"\nRunning inference on {num_cases} cases...\n")

    # ---------------------------------------------------------------------
    # Inference loop
    # ---------------------------------------------------------------------

    for i in range(num_cases):
        batch = next(test_gen)

        case_name = batch["cases"][0]
        data = maybe_to_torch(batch["data"]).to(trainer.device)

        # -------------------------------------------------------------
        # Handle target format
        # -------------------------------------------------------------

        if "target" in batch:
            target = [
                {k: v.to(trainer.device) for k, v in t.items()}
                for t in batch["target"]
            ]
        else:
            target = [
                {
                    "labels": labels.to(trainer.device),
                    "masks": masks.to(trainer.device),
                }
                for labels, masks in zip(batch["labels"], batch["masks"])
            ]

        # -------------------------------------------------------------
        # Forward pass
        # -------------------------------------------------------------

        with torch.no_grad():
            output = trainer.network(data)

        pred_seg = get_pred_seg(output, num_classes, trainer.device)

        B, D, H, W = pred_seg.shape

        # -------------------------------------------------------------
        # Brain mask: remove predictions outside skull
        # -------------------------------------------------------------
        # Assumes T1 is channel 0.
        # If you run a single modality that is not T1, adjust this logic.

        t1_channel = data[0, 0]
        brain_mask = (t1_channel > 0.01).to(trainer.device)

        for b in range(B):
            pred_seg[b][~brain_mask] = 0

        # -------------------------------------------------------------
        # Build dense ground-truth segmentation map
        # -------------------------------------------------------------

        gt_seg = build_gt_seg(
            batch=batch,
            target=target,
            batch_size=B,
            depth=D,
            height=H,
            width=W,
            device=trainer.device,
        )

        # -------------------------------------------------------------
        # Select top 3 slices by tumor content
        # -------------------------------------------------------------

        gt_slice = gt_seg[0].cpu()
        pred_slice = pred_seg[0].cpu()

        tumor_per_slice = gt_slice.bool().sum(dim=(1, 2))
        sorted_slices = tumor_per_slice.argsort(descending=True)

        print(f"\n[{i + 1}/{num_cases}] {case_name}")

        for rank in range(min(3, D)):
            best_d = sorted_slices[rank].item()

            # Stop if no tumor exists in this slice
            if tumor_per_slice[best_d].item() == 0:
                break

            pred_2d = pred_slice[best_d].numpy()
            gt_2d = gt_slice[best_d].numpy()

            # T1ce is usually channel 1 for multimodal input:
            # modality = ("t1", "t1ce", "t2", "flair")
            #
            # If only one modality is used, fall back to channel 0.
            image_channel_idx = 1 if data.shape[1] > 1 else 0
            image_2d = data[0, image_channel_idx, best_d].cpu().numpy()

            save_path = save_prediction_figure(
                image_2d=image_2d,
                gt_2d=gt_2d,
                pred_2d=pred_2d,
                case_name=case_name,
                model_name=model,
                rank=rank,
                slice_idx=best_d,
                preds_dir=preds_dir,
                label_legend=label_legend,
                show=not args.no_show,
            )

            print(f"  rank {rank + 1} saved → {save_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()