import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from scipy.optimize import linear_sum_assignment

class HungarianMatcher3D(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def compute_cls_loss(self, inputs, targets):
        """ Classification loss (NLL)
            implemented in compute_loss()
        """
        raise NotImplementedError 

        
    def compute_dice_loss(self, inputs, targets):
        """ mask dice loss
            inputs (B*K, C, H, W)
            target (B*K, D, H, W)
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        num_masks = len(inputs)

        numerator = 2 * (inputs * targets).sum()
        denominator = inputs.sum() + targets.sum()
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


    def compute_ce_loss(self, inputs, targets):
        """mask ce loss"""
        num_masks = len(inputs)
        loss = F.binary_cross_entropy_with_logits(inputs.flatten(1), targets.flatten(1), reduction="none")
        loss = loss.mean(1).sum() / num_masks
        return loss

    def compute_dice(self, inputs, targets):
        """ output (N_q, C, H, W)
            target (K, D, H, W)
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        hw = inputs.shape[1]
        targets = targets.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs/ hw , targets)
        denominator = inputs.sum(-1)[:, None] / hw + targets.sum(-1)[None, :] / hw
        loss = 1 - (numerator + 1/hw ) / (denominator + 1/hw )
        return loss # [N_q, K]


    def compute_ce(self, inputs, targets):
        """ output (N_q, C, H, W)
            target (K, D, H, W)
            return (N_q, K)
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )

        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos / hw, targets) +  torch.einsum("nc,mc->nm", neg / hw, (1 - targets))

        return loss

        # target_onehot = torch.zeros_like(output, device=output.device)
        # target_onehot.scatter_(1, target.long(), 1)
        # assert (torch.argmax(target_onehot, dim=1) == target[:, 0].long()).all()
        # ce_loss = F.binary_cross_entropy_with_logits(output, target_onehot)
        # return ce_loss


    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching for single aux, outputs: (b, q, d, h, w)"""
        """suppose each crop must contain foreground class"""
        bs, num_queries =  outputs["pred_logits"].shape[:2]
        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1) # [num_queries, num_classes+1]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            tgt_mask = targets[b]["masks"].to(out_mask) # [K, D, H, W], K is number of classes shown in this image, and K < n_class

            # target_onehot = torch.zeros_like(tgt_mask, device=out_mask.device)
            # target_onehot.scatter_(1, targets.long(), 1)
            # ── add this diagnostic ────────────────────────────
            # print(f"batch {b}:")
            # print(f"  num_queries: {num_queries}")
            # print(f"  tgt_ids:     {tgt_ids}")
            # print(f"  tgt_mask:    {tgt_mask.shape}")
            # print(f"  out_prob:    {out_prob.shape}")
            # print(f"  out_mask:    {out_mask.shape}")
            # print(f"  tgt_mask sum per organ: {tgt_mask.sum(dim=[1,2,3])}")
        # ──────────────────────────────────────────────────

            cost_class = -out_prob[:, tgt_ids] # [num_queries, K]

            with autocast(device_type='cuda'):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()

                out_mask = torch.clamp(out_mask, min=-10.0, max=10.0) #clamped mask value

                cost_dice = self.compute_dice(out_mask, tgt_mask)
                cost_mask = self.compute_ce(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )

            C = C.reshape(num_queries, -1).cpu() # (num_queries, K)
            # print(f"  C shape: {C.shape}")
            # print(f"  C range: {C.min():.3f} to {C.max():.3f}")
            # print(f"  C has nan: {torch.isnan(C).any()}")
            # print(f"  C has inf: {torch.isinf(C).any()}")

            # linear_sum_assignment return a tuple of two arrays: row_ind, col_ind, the length of array is min(N_q, K)
            # The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum()

            indices.append(linear_sum_assignment(C))

        final_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

        return final_indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
def compute_loss_hungarian(outputs, targets, idx, matcher, num_classes, point_rend=False, num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, no_object_weight=None, cost_weight=[2,5,5]):
    """output is a dict only contain keys ['pred_masks', 'pred_logits'] """
    # outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}

    indices = matcher(outputs, targets) 
    src_idx = matcher._get_src_permutation_idx(indices) # return a tuple of (batch_idx, src_idx)
    tgt_idx = matcher._get_tgt_permutation_idx(indices) # return a tuple of (batch_idx, tgt_idx)
    assert len(tgt_idx[0]) ==  sum([len(t["masks"]) for t in targets]) # verify that all masks of (K1, K2, ..) are used
    
    # step2 : compute mask loss
    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx] # [len(src_idx[0]), D, H, W] -> (K1+K2+..., D, H, W) 
    target_masks = torch.cat([t["masks"] for t in targets], dim=0) # (K1+K2+..., D, H, W) actually
    src_masks = src_masks[:, None] # [K..., 1, D, H, W]
    target_masks = target_masks[:, None]

    if point_rend: # only calculate hard example
        with torch.no_grad():
            # num_points=12544 config in cityscapes

            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            ) # [K, num_points=12544, 3]

            point_labels = point_sample_3d(
                target_masks.float(),
                point_coords.float(),
                align_corners=False,
            ).squeeze(1) # [K, 12544]

        point_logits = point_sample_3d(
                src_masks.float(),
                point_coords.float(),
                align_corners=False,
        ).squeeze(1) # [K, 12544]

        src_masks, target_masks = point_logits, point_labels

    loss_mask_ce = matcher.compute_ce_loss(src_masks, target_masks) 
    loss_mask_dice = matcher.compute_dice_loss(src_masks, target_masks)
    
    # step3: compute class loss
    src_logits = outputs["pred_logits"].float() # (B, num_query, num_class+1)
    target_classes_o = torch.cat([t["labels"] for t in targets], dim=0) # (K1+K2+, )
    target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        ) # (B, num_query, num_class+1)
    target_classes[src_idx] = target_classes_o


    if no_object_weight is not None:
        empty_weight = torch.ones(num_classes + 1).to(src_logits.device)
        empty_weight[-1] = no_object_weight
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    else:
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

    loss = (cost_weight[0]/10)*loss_cls + (cost_weight[1]/10)*loss_mask_ce + (cost_weight[2]/10)*loss_mask_dice # 2:5:5, like hungarian matching
    # print("idx {}, loss {}, loss_cls {}, loss_mask_ce {}, loss_mask_dice {}".format(idx, loss, loss_cls, loss_mask_ce, loss_mask_dice))
    return loss


def point_sample_3d(input, point_coords, **kwargs):
    """
    from detectron2.projects.point_rend.point_features
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Hgrid, Wgrid, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Dgrid, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2) # why
    
    # point_coords should be (N, D, H, W, 3)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        output = output.squeeze(3).squeeze(3)

    return output


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# implemented!
def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    n_dim = 3
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio) # 12544 * 3, oversampled
    point_coords = torch.rand(num_boxes, num_sampled, n_dim, device=coarse_logits.device) # (K, 37632, 3); uniform dist [0, 1)
    point_logits = point_sample_3d(coarse_logits, point_coords, align_corners=False) # (K, 1, 37632)

    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points) # 9408
    
    num_random_points = num_points - num_uncertain_points # 3136
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None] # [K, 9408]

    point_coords = point_coords.view(-1, n_dim)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, n_dim
    )  # [K, 9408, 3]
    
    if num_random_points > 0:
        # from detectron2.layers import cat
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, n_dim, device=coarse_logits.device),
            ],
            dim=1,
        ) # [K, 12544, 3]

    return point_coords



def cnn_compute_ce_loss(seg_logits, gt_seg):
    """ce loss"""
    # weight rare classes higher to combat imbalance
    # background=1, tumor classes=10
    num_classes = seg_logits.shape[1]
    weights = torch.ones(num_classes, device=seg_logits.device)
    weights[1:] = 10.0   # upweight all tumor classes
    
    loss = F.cross_entropy(seg_logits, gt_seg, weight=weights)
    return loss

def cnn_compute_dice_loss(seg_logits, gt_seg, NUM_CLASSES):
        """
        converts ground truth segmentation to one-hot 
        so shapes match seg
        """
        assert isinstance(gt_seg, torch.Tensor), f"gt_seg must be tensor, got {type(gt_seg)}"
    
        gt_one_hot = F.one_hot(gt_seg, num_classes=NUM_CLASSES) # (B,D,H,W,C)
        gt_one_hot = gt_one_hot.permute(0,4,1,2,3).float()        # (B,C,D,H,W)

        seg_soft = F.softmax(seg_logits, dim=1)

        loss_dice = 0.0

        for c in range(1, NUM_CLASSES):
            pred = seg_soft[:, c].flatten()
            gt = gt_one_hot[:, c].flatten()
            num = 2 * (pred * gt).sum()
            den = pred.sum() + gt.sum() + 1e-8
            loss_dice += 1 - (num +1) / (den + 1)
        loss_dice /= (NUM_CLASSES - 1)
        return loss_dice
   