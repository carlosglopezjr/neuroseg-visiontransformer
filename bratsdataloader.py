import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from resize_right import resize
import interp_methods

"""
BraTSDataset loader that finds all cases from directory and stacks modalities as classes
so they can be interpreted as channels

"""

# ── Step 1: BraTS Dataset ──────────────────────────────────────
class BraTSDataset(Dataset):

    BRATS_CLASSES = {
        0: 'background',
        1: 'necrotic_core',
        2: 'edema',
        4: 'enhancing_tumor',
    }
    # remap label 4 → 3 for contiguous indexing
    LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    NUM_CLASSES = 4   # background + 3 tumor regions

    def __init__(self,
                 mode,
                 data_dir,
                 patch_size,
                 modalities=('t1', 't1ce', 't2', 'flair'),
                 augment=False):

        self.mode = mode
        self.data_dir   = data_dir
        self.patch_size = patch_size
        self.modalities = modalities
        self.augment    = augment

        # find all case folders in your directory for later use
        self.cases = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        print(f"found {len(self.cases)} BraTS cases")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # Allows indexing of cases made during intialization
        case_dir  = os.path.join(self.data_dir, self.cases[idx])
        case_name = self.cases[idx]

        # load modalities
        # BraTS has 4 MRI channels
        volumes = [None] * len(self.modalities)
        
        for i,mod in enumerate(self.modalities):
            path = os.path.join(case_dir, f"{case_name}_{mod}.nii.gz")
            vol  = nib.load(path).get_fdata()          # (H, W, D)
            vol  = vol.transpose(2, 0, 1)              # → (D, H, W)
            vol  = self._normalize(vol)                # normalize per modality
            volumes[i] = vol
            

        # stack modalities → (C, D, H, W)
        #so that each modality can be interpreted as a channel
        image = np.stack(volumes, axis=0).astype(np.float32)

        # ── load segmentation ────────────
        #data contatains 3D grid of integers, reorders ground truth lables
        #so our model can correctly interpret the masks
        seg_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
        seg = nib.load(seg_path).get_fdata()           # (H, W, D)
        seg = seg.transpose(2, 0, 1)                   # → (D, H, W)

        # remap label 4 → 3 for contiguous indexing
        seg_remapped = np.zeros_like(seg)
        for orig, new in self.LABEL_MAP.items():
            seg_remapped[seg == orig] = new
        seg = seg_remapped.astype(np.int64)

        # ── resize to patch_size using resize_right ────────────
        image_t = torch.tensor(image)    # (C, D, H, W)
        seg_t   = torch.tensor(seg).float().unsqueeze(0)  # (1, D, H, W)

        pd, ph, pw = self.patch_size

        # resize image — antialiasing for downsampling
        image_resized = resize(
            image_t,
            out_shape=(len(self.modalities), pd, ph, pw),
            interp_method=interp_methods.cubic,
            antialiasing=True,
        )                                              # (C, pd, ph, pw)

        # resize seg — nearest neighbor to preserve integer labels
        seg_resized = resize(
            seg_t,
            out_shape=(1, pd, ph, pw),
            interp_method=interp_methods.linear,
            antialiasing=False,
        ).squeeze(0).round().long()                    # (pd, ph, pw)

        # ── augmentation ───────────────────────────────────────
        if self.augment:
            image_resized, seg_resized = self._augment(image_resized, seg_resized)

        #Preallocate memory
        masks  = [torch.zeros(pd,ph,pw) for _ in range(self.NUM_CLASSES)]
        labels = [0] * self.NUM_CLASSES
    
        # ── build targets dict ─────────────────────────────────
        # one binary mask per tumor class present in this crop
        #Encoder Decoder Full TransUnet setup
        for class_idx in range(1, self.NUM_CLASSES):   # skip background
            binary_mask = (seg_resized == class_idx).float()
            if binary_mask.sum() > 0:                  # only include if present
                masks[class_idx] = binary_mask
                labels[class_idx] = class_idx
   

        masks  = torch.stack(masks)                    # (N_organs, D, H, W)
        labels = torch.tensor(labels, dtype=torch.long)

        #CNN-only and Encoder-only (CE+ Diceloss)
        if self.mode in ("CNN", "ENCODER"): 
            return [case_name, image_resized,labels, masks]
        else: #input should be "Decoder" or "FullTransUNet", mask classification configs - Hungarian
            return {
                'data':   image_resized,                   # (C, D, H, W)
                'target': [{'labels': labels, 'masks': masks}],
                'case':   case_name,
            }


    def _normalize(self, vol):
        """z-score normalization per volume — standard for MRI"""
        mask = vol > 0 # brain region only (ignore background zeros)
        if mask.sum() == 0:
            return vol
        mean = vol[mask].mean()
        std  = vol[mask].std() + 1e-8
        vol  = (vol - mean) / std
        return vol


    def _augment(self, image, seg):
        """random flips, allows us to get more out of our data"""
        # random flip along depth,
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[1])
            seg   = torch.flip(seg,   dims=[0])
        # random flip along height
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            seg   = torch.flip(seg,   dims=[1])
        return image, seg


# ── Step 2: Collate function ───────────────────────────────────
def brats_collate(batch):
    """
    custom collate — one scan might have one class, another might
    have 3. Cant stack tensors of different sizes,keeps as python list
    """
    #Might have to change depending on expected input
    if isinstance(batch[0], dict):
        images  = torch.stack([item['data'] for item in batch])   # (B, C, D, H, W)
        targets = [item['target'][0] for item in batch]           # list of dicts
        cases   = [item['case'] for item in batch]
        return {
            'data':   images,
            'target': targets,
            'cases':  cases,
        }
    if isinstance(batch[0], list):
        cases   = [item[0] for item in batch]
        images  = torch.stack([item[1] for item in batch])  
        labels = [item[2] for item in batch]           
        masks   = [item[3] for item in batch]
        return {
            'data':   images,
            'cases':  cases,
            'labels': labels,
            'masks' : masks,
        }



# ── Step 3: Generator wrapper ──────────────────────────────────
class InfiniteDataLoader:
    """
    wraps dataLoader to yield batches infinitely.
    matches the next(data_generator) pattern from network_trainer.py
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator   = iter(dataloader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def next(self):
        return self.__next__()


# ── Step 4: Put it all together ────────────────────────────────
def create_brats_dataloaders(mode,
                              data_dir,
                              modalities,
                              patch_size=(40, 224, 224),
                              batch_size=2,
                              train_ratio=0.8,
                              num_workers=2):
    """split dataset into train/val and create infinite generators"""

    dataset = BraTSDataset(
        mode = mode,
        data_dir=data_dir,
        patch_size=patch_size,
        modalities=modalities,
        augment=True,
    )

    # train/val split
    n_train = int(len(dataset) * train_ratio)
    n_val   = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(12345)
    )
    # turn off augmentation for val
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=brats_collate,
        pin_memory=True,
    )


    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=brats_collate,
        pin_memory=True,
    )
   

    return InfiniteDataLoader(train_loader), InfiniteDataLoader(val_loader)