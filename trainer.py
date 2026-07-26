
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from transunet3d_model import TransUNet
from hungarian3d import HungarianMatcher3D, compute_loss_hungarian, cnn_compute_ce_loss, cnn_compute_dice_loss
import datetime
import time


# ── from network_trainer.py ─────────────
#taken from becks code
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def maybe_to_torch(d):
    """
    converts numpy arrays to floats
    """
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d
# ──────────────────────

class TransUNetTrainer:

    def __init__(self,
                output_folder='./checkpoints',
                initial_lr=1e-4,         # AdamW default (paper uses 1e-4)
                max_num_epochs=1000,
                num_batches_per_epoch=250,
                num_val_batches_per_epoch=50,
                fp16=True,               # mixed precision — halves memory
                save_every=50,
                num_classes=10,
                patch_size=(224, 224, 144),
                batch_size=2,
                plot_title = "Model",
                ):

        self.output_folder         = output_folder
        self.initial_lr            = initial_lr
        self.max_num_epochs        = max_num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_val_batches_per_epoch = num_val_batches_per_epoch
        self.fp16                  = fp16
        self.save_every            = save_every
        self.num_classes           = num_classes
        self.patch_size            = patch_size
        self.batch_size            = batch_size
        self.epoch                 = 0
        self.plot_title            = plot_title + " Loss"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"training on: {self.device}")

        # loss tracking
        self.all_tr_losses  = [None] * self.max_num_epochs
        self.all_val_losses = [None] * self.max_num_epochs

        os.makedirs(output_folder, exist_ok=True)

    def initialize(self):
        """build model, optimizer, matcher, scaler"""

        # ── model ──────────────────────────────────────────────
        self.network = TransUNet(
            input_channels=1,
            base_num_features=32,
            num_classes=self.num_classes,
            num_pool=5,
            pool_op_kernel_sizes=[[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]], #might remove if causes issues
            conv_kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
            patch_size=list(self.patch_size),
            is_max=True,
            is_max_cls=True,
            is_max_ds=True,
            is_max_hungarian=True,
            mw=1.0,
            deep_supervision=False,
            is_max_bottleneck_transformer=True,
            vit_depth=12,               # paper uses 12 for best results
        ).to(self.device)

        # ── optimizer — AdamW matches paper ──────────────
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-4,
        )

        # ── Hungarian matcher ─────────────────────
        self.matcher = HungarianMatcher3D(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
        )

        # ── mixed precision scaler ──────────────────────────────
        # from network_trainer.py _maybe_init_amp pattern
        self.amp_grad_scaler = GradScaler(device='cuda') if self.fp16 else None

        print("model initialized")
        total = sum(p.numel() for p in self.network.parameters())
        print(f"total parameters: {total:,}")

    def run_iteration(self, data_generator, do_backprop=True):
        """
        single training or validation step.
        pattern taken directly from network_trainer.py run_iteration()

        note: do_backprop set to false will be used during validation run
        forward and backward pass happen but weights are not updated
        """
        # get batch from generator
        data_dict = next(data_generator)
        data   = maybe_to_torch(data_dict['data']).to(self.device)
        
        if 'target' in data_dict:
            #Transformer decoder mode
            target = [{k: v.to(self.device) for k, v in t.items()} for t in data_dict['target']]
        
        else:
            #CNN mode
            target = [{
                'labels': l.to(self.device),
                'masks' : m.to(self.device),
            } for l, m in zip(data_dict['labels'], data_dict['masks'])]

        # move targets to device
        

        self.optimizer.zero_grad()

        # ── forward + loss ─────────────────────────────────────
        if self.fp16:
            # mixed precision — from network_trainer.py lines 837-846
            with autocast(device_type='cuda'):
                output = self.network(data)
                del data
                        # check for NaN
                #for diagnosing erorr when using TransformerDecoder
                # if torch.isnan(output['pred_masks']).any():
                #     print(f"NaN in pred_masks at epoch {self.epoch}")
                # if torch.isnan(output['pred_logits']).any():
                #     print(f"NaN in pred_logits at epoch {self.epoch}")
                # print(f"pred_masks range: {output['pred_masks'].min():.3f} to {output['pred_masks'].max():.3f}")

                #CNN-Only mode
                if not self.network.is_max:
                    seg = output[0] if isinstance(output, tuple) else output

                    B, C, D, H, W = seg.shape
                    gt_seg = torch.zeros(B, D, H, W,
                                        dtype=torch.long,
                                        device=self.device)
                    for b, t in enumerate(target):
                        for mask, label in zip(t['masks'], t['labels']):
                            gt_seg[b][mask.bool()] = label
                    l =cnn_compute_ce_loss(seg, gt_seg) + cnn_compute_dice_loss(seg,gt_seg,self.num_classes)
                else:    
                    l = compute_loss_hungarian(
                        outputs=output,
                        targets=target,
                        idx=self.epoch,
                        matcher=self.matcher,
                        num_classes=self.num_classes,
                        cost_weight=[2, 5, 5],
                    )
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data

            if not self.network.is_max:
                #CN-only mode
                seg = output[0] if isinstance(output, tuple) else output

                B, C, D, H, W = seg.shape
                gt_seg = torch.zeros(B, D, H, W,
                                    dtype=torch.long,
                                    device=self.device)
                for b, t in enumerate(target):
                        for mask, label in zip(t['masks'], t['labels']):
                            gt_seg[b][mask.bool()] = label
                l =cnn_compute_ce_loss(seg, gt_seg) + cnn_compute_dice_loss(seg,gt_seg,self.num_classes)
            else:
                #set to batch dice loss
                l = compute_loss_hungarian(
                    outputs=output,
                    targets=target,
                    idx=self.epoch,
                    matcher=self.matcher,
                    num_classes=self.num_classes,
                    cost_weight=[2, 5, 5],
                )
            if do_backprop:
                l.backward()
                self.optimizer.step()

        del target
        return l.detach().cpu().numpy()

    def run_training(self, tr_gen, val_gen):
        """
        main training loop.
        structure taken from network_trainer.py run_training() lines 484-554
        """


        # preserve or extend loss history 
        if self.all_tr_losses is None or not isinstance(self.all_tr_losses, list):
            # fresh start — initialize from scratch
            self.all_tr_losses  = [None] * self.max_num_epochs
            self.all_val_losses = [None] * self.max_num_epochs

        elif len(self.all_tr_losses) < self.max_num_epochs:
            # extend — resuming with more epochs than originally planned
            extra = self.max_num_epochs - len(self.all_tr_losses)
            self.all_tr_losses  = self.all_tr_losses  + [None] * extra
            self.all_val_losses = self.all_val_losses + [None] * extra
            # ↑ preserves epochs 0-18, adds None slots for 19+

        elif len(self.all_tr_losses) > self.max_num_epochs:
            # truncate — resuming with fewer epochs (unlikely but safe)
            self.all_tr_losses  = self.all_tr_losses[:self.max_num_epochs]
            self.all_val_losses = self.all_val_losses[:self.max_num_epochs]
        # else: lengths match exactly — no change needed

        training_start = time.time()
        epoch_times = []


        while self.epoch < self.max_num_epochs:
            epoch_start = time.time()
            print(f"\n── epoch {self.epoch} ──────────────────────")

            # ── train -------------------------------
            self.network.train()
            train_losses = [None] * self.num_batches_per_epoch
            for i in range(self.num_batches_per_epoch):
                l = self.run_iteration(tr_gen, do_backprop=True)
                train_losses[i]= l

            self.all_tr_losses[self.epoch] = np.mean(train_losses) #self.all_tr_losses should be len(self.max_num_epochs)
            print(f"train loss: {self.all_tr_losses[self.epoch]:.4f}")

            # ── validate -------------------------------
            self.network.eval()
            val_losses = [None] * self.num_val_batches_per_epoch
            with torch.no_grad():
                for i in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(val_gen, do_backprop=False)
                    val_losses[i] = l
            self.all_val_losses[self.epoch] = np.mean(val_losses)
            print(f"val loss:   {self.all_val_losses[self.epoch]:.4f}")

            # ── timing ─────────────────────────────────
            epoch_time = time.time() - epoch_start       
            epoch_times.append(epoch_time)               

            epochs_done      = self.epoch + 1                                    
            epochs_remaining = self.max_num_epochs - epochs_done                 
            avg_epoch_time   = np.mean(epoch_times)                              
            eta_seconds      = epochs_remaining * avg_epoch_time

            # format ETA nicely                                                   
            if eta_seconds > 3600:                                                
                eta_str = f"{eta_seconds/3600:.1f}h"                             
            else:                                                                 
                eta_str = f"{eta_seconds/60:.0f}min"                             

            print(f"epoch time: {epoch_time:.1f}s  |  "                         
                f"avg: {avg_epoch_time:.1f}s  |  "                            
                f"ETA: {eta_str}  |  "                                       
                f"elapsed: {(time.time()-training_start)/60:.1f}min") 

            # ── update learning rate — poly_lr from network_trainer.py
            new_lr = poly_lr(self.epoch, self.max_num_epochs, self.initial_lr, 0.9)
            self.optimizer.param_groups[0]['lr'] = new_lr
            print(f"lr:         {new_lr:.6f}")

            # ── checkpoint -------------------------------
            if self.epoch % self.save_every == (self.save_every - 1):
                self.save_checkpoint(f'{self.output_folder}/checkpoint_ep{self.epoch}.pt')

            self.epoch += 1

        # format ETA nicely                                                  
        if eta_seconds > 3600:                                                
            eta_str = f"{eta_seconds/3600:.1f}h"                            
        else:                                                                
            eta_str = f"{eta_seconds/60:.0f}min"                             

        total_time = time.time() - training_start
        print(f"\n── training complete ───────────────────────────")
        print(f"total time:     {total_time/3600:.2f}h ({total_time/60:.0f}min)")
        print(f"avg epoch time: {np.mean(epoch_times):.1f}s")
        print(f"fastest epoch:  {min(epoch_times):.1f}s")
        print(f"slowest epoch:  {max(epoch_times):.1f}s")

        # save final
        self.save_checkpoint(f'{self.output_folder}/model_final.pt')
        print("training complete")

        #run final visualization
        self.visualize_loss()

    def visualize_loss(self):

        tr_pairs  = [(i,l) for i,l in enumerate(self.all_tr_losses)  if l is not None]
        val_pairs = [(i,l) for i, l in enumerate(self.all_val_losses) if l is not None]

        epochs     = [p[0] for p in tr_pairs]
        tr_losses  = [p[1] for p in tr_pairs]
        val_losses = [p[1] for p in val_pairs]

        avg_tr  = np.mean(tr_losses)
        avg_val = np.mean(val_losses)
        min_tr  = min(tr_losses)
        min_val = min(val_losses)

        fig, (ax_plot, ax_table) = plt.subplots(
            2, 1,
            figsize=(10, 9),
            gridspec_kw={'height_ratios': [4, 1]}
        )

        # ── loss curves -------------------------------
        ax_plot.plot(epochs, tr_losses,  label='train loss',
                    color='blue', linewidth=2)
        ax_plot.plot(epochs, val_losses, label='val loss',
                    color='red',  linewidth=2, alpha=0.7)
        ax_plot.axhline(y=avg_tr,  color='blue', linestyle='--',
                        linewidth=1.2, alpha=0.6,
                        label=f'avg train: {avg_tr:.4f}')
        ax_plot.axhline(y=avg_val, color='red',  linestyle='--',
                        linewidth=1.2, alpha=0.6,
                        label=f'avg val:   {avg_val:.4f}')

        ax_plot.set_xlabel('epoch')
        ax_plot.set_ylabel('loss (raw value)')
        ax_plot.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.3f}')
        )
        ax_plot.set_title(self.plot_title)
        ax_plot.legend(loc='upper right')
        ax_plot.grid(True, alpha=0.3)

        # ── summary table -------------------------------
        ax_table.axis('off')

        table_data = [
            ['',      'avg',            'min',            'final'],
            ['train', f'{avg_tr:.4f}',  f'{min_tr:.4f}',  f'{tr_losses[-1]:.4f}'],
            ['val',   f'{avg_val:.4f}', f'{min_val:.4f}', f'{val_losses[-1]:.4f}'],
        ]

        table = ax_table.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        for col in range(4):
            table[0, col].set_facecolor('#DDEEFF')
            table[0, col].set_text_props(fontweight='bold')
            table[1, col].set_facecolor('#EEF4FF')
            table[2, col].set_facecolor('#FFF0F0')

        plt.tight_layout(pad=2.0)

        save_path = os.path.join(self.output_folder, self.plot_title + '.jpg')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"saved to {save_path}")


    def save_checkpoint(self, path):
        """pattern from network_trainer.py save_checkpoint"""
        torch.save({
            'epoch':                    self.epoch,
            'model_state_dict':         self.network.state_dict(),
            'optimizer_state_dict':     self.optimizer.state_dict(),
            'amp_grad_scaler':          self.amp_grad_scaler.state_dict() if self.fp16 else None,
            'all_tr_losses':            self.all_tr_losses,
            'all_val_losses':           self.all_val_losses,
        }, path)
        print(f"checkpoint saved: {path}")


    def load_checkpoint(self, path):
        """resume training from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.fp16 and checkpoint['amp_grad_scaler'] is not None:
            scaler_state = checkpoint['amp_grad_scaler']
            if len(scaler_state) > 0:  # only load if non-empty
                self.amp_grad_scaler.load_state_dict(scaler_state)
        self.epoch          = checkpoint['epoch']
        self.all_tr_losses  = checkpoint['all_tr_losses']
        self.all_val_losses = checkpoint['all_val_losses']
        print(f"resumed from epoch {self.epoch}")


    
    def evaluate_dice(self, val_gen,
                    regions=('ET','TC','WT')):
        """
        regions: which BraTs regions to evaluate
            ('ET',)     - enhancing tumor only
            ('WT',)     - whole tumor
            ('ET','TC','WT')     - all three regions
        """
        self.network.eval()
        scores = {r: [] for r in regions}

        #for each batch in val_gen...
        with torch.no_grad():
                for _ in range(self.num_val_batches_per_epoch):
            #         l = self.run_iteration(val_gen, do_backprop=False)
            #         val_losses[i] = l
            # self.all_val_losses[self.epoch] = np.mean(val_losses)
            # print(f"val loss:   {self.all_val_losses[self.epoch]:.4f}") 

                    #get the data and the targets...
                    data_dict = next(val_gen)
                    data   = maybe_to_torch(data_dict['data']).to(self.device)
                    
                    if 'target' in data_dict:
                        #Transformer decoder mode
                        target = [{k: v.to(self.device) for k, v in t.items()} for t in data_dict['target']]
                    
                    else:
                        #CNN mode
                        target = [{
                            'labels': l.to(self.device),
                            'masks' : m.to(self.device),
                        } for l, m in zip(data_dict['labels'], data_dict['masks'])]

            
                    output = self.network(data)

                    if isinstance(output, dict):
                        pred_masks_sig = output['pred_masks'].sigmoid()
                        pred_logits    = output['pred_logits']
                        B = pred_masks_sig.shape[0]
                        pred_seg = torch.zeros(B, *pred_masks_sig.shape[2:],
                                                dtype=torch.long,
                                                device=self.device)
                        for b in range(B):
                            logits          = pred_logits[b]
                            masks           = pred_masks_sig[b]
                            classes         = logits.argmax(dim=-1)
                            query_per_voxel = masks.argmax(dim=0)
                            voxel_classes   = classes[query_per_voxel]
                            voxel_classes[voxel_classes >= self.num_classes] = 0  # ← no-object fix
                            pred_seg[b]     = voxel_classes
                        D, H, W = pred_seg.shape[1:]

                    else:
                        seg      = output[0] if isinstance(output, tuple) else output
                        pred_seg = seg.argmax(dim=1)
                        B, C, D, H, W = seg.shape  # ← keep C here since it's CNN output

                        # B, D, H, W always valid after both paths
                    B, D, H, W = pred_seg.shape

                    #Buid integer label map-----------------
                    gt_seg = torch.zeros(B, D, H, W,
                                        dtype=torch.long,
                                        device=self.device)
                    for b, t in enumerate(target):
                        for mask, label in zip(t['masks'], t['labels']):
                            gt_seg[b][mask.bool()] = label

                    pred_et = (gt_seg == 3).float()
                    gt_et   = (gt_seg == 3).float()
                    num = 2 * (pred_et * gt_et).sum()
                    den = pred_et.sum() + gt_et.sum() + 1e-8
                    print(f"perfect ET Dice: {(num/den).item():.4f}")   # should be 1.0000

                    # feed all-zero prediction — should give Dice = 0
                    pred_zero = torch.zeros_like(gt_seg)
                    pred_et   = (pred_zero == 3).float()
                    num = 2 * (pred_et * gt_et).sum()
                    den = pred_et.sum() + gt_et.sum() + 1e-8
                    print(f"zero pred Dice:  {(num/den).item():.4f}")   # should be 0.0000

                    #Compute dice scores for ET, TC and WT

                    #Uses brats remapping 
                    if "ET" in regions:
                        pred_et = (pred_seg == 3).float()
                        gt_et = (gt_seg == 3).float()
                        num = 2 * (pred_et * gt_et).sum()
                        den = pred_et.sum() + gt_et.sum() +1e-8
                        scores['ET'].append((num/den).item())
                    if "TC" in regions:
                        pred_tc = ((pred_seg == 2) | (pred_seg == 3)).float()
                        gt_tc = ((gt_seg == 2) | (gt_seg == 3)).float()
                        num = 2 * (pred_tc * gt_tc).sum()
                        den = pred_tc.sum() + gt_tc.sum() +1e-8
                        scores['TC'].append((num/den).item())
                    if "WT" in regions:
                        pred_wt = (pred_seg >= 1).float()
                        gt_wt = (gt_seg >= 1).float()
                        num = 2 * (pred_wt * gt_wt).sum()
                        den = pred_wt.sum() + gt_wt.sum() +1e-8
                        scores['WT'].append((num/den).item())

        results = {r: np.mean(v) for r,v in scores.items()}
        results['mean'] = np.mean(list(results.values()))

        print("Dice scores---------------------")
        for r,v in results.items():
            print(f"{r}: {v:.4f} ({v*100:.1f}%)")

        return results
    def save_dice_results(self,results, model_name, modality, output_dir):
        """
        saves dice scores to a txt file
        results:    dict from evaluate_dice — {'ET': float, 'TC': float, 'WT': float, 'mean': float}
        model_name: string — e.g. 'CNN', 'ENCODER-ONLY'
        modality:   tuple  — e.g. ('t1', 't1ce')
        output_dir: where to save
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"dice_{model_name}_{timestamp}.txt"
        save_path = os.path.join(output_dir, filename)

        with open(save_path, 'w') as f:
            f.write(f"TransUNet Dice Score Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"model:      {model_name}\n")
            f.write(f"modality:   {modality}\n")
            f.write(f"timestamp:  {timestamp}\n")
            f.write(f"{'='*40}\n\n")
            
            for region, score in results.items():
                if region != 'mean':
                    f.write(f"{region}:    {score:.4f}  ({score*100:.1f}%)\n")
            
            f.write(f"\nmean:  {results['mean']:.4f}  ({results['mean']*100:.1f}%)\n")

        print(f"dice scores saved to {save_path}")
        return save_path


    # your training code here
    # trainer.run_training()
