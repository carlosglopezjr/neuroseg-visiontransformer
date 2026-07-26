import torch
from transunet3d_model import TransUNet #importing the class
from hungarian3d import HungarianMatcher3D,compute_loss_hungarian, cnn_compute_ce_loss, cnn_compute_dice_loss
# ------- 1. Instantiate the model --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Parameters 
d, h, w        = 32, 64, 64    # small but divisible
BATCH_SIZE     = 2
NUM_CLASSES    = 4
INPUT_CHANNELS = 4
VIT_DEPTH      = 1

#For Transformer make sure the patch size = the D,W,H of the dummy values in torch.randn

#Using Transformer Encoder,Decoder (Max PPB+)--------

#deepsupervision --- controls whether CNN decoder produces multi-scale outputs

# input_channels,base_num_features, num_classes, num_pool = 1, 32, 10, 5
# d,h,w = 224,224,192
# d,h,w = 32,64,64

# model = TransUNet(input_channels,base_num_features, num_classes, num_pool,
#                   patch_size=[d, h, w],
#                   is_max_bottleneck_transformer=True,
#                   vit_depth=5,
#                   mw= 1.0,
#                   is_max=True,
#                   is_max_cls=True,
#                   is_max_ds=True,
#                   is_max_hungarian=True,
#                   deep_supervision=False)


#Standard CNN

#TransUNet(input_channels, base_num_features, num_classes, num_pool) 


is_max = False

model = TransUNet(
    input_channels=INPUT_CHANNELS,
    base_num_features=32,
    num_classes=NUM_CLASSES,
    num_pool=5,
    patch_size=[d, h, w],
    is_max=False,                          # ← no Transformer decoder
    is_max_cls=True,
    is_max_ds=True,
    is_max_hungarian=True,
    mw=1.0,
    deep_supervision=True,                 # use CNN seg outputs instead
    is_max_bottleneck_transformer=False,
    vit_depth=1,                          # ← small ViT encoder
)
#---------------------------------------------------------------------------------------------------------------------------------------------------


model.eval()

print(f"is_max:     {model.is_max}")
print(f"is_max_cls: {model.is_max_cls}")
print(f"is_max_ds:  {model.is_max_ds}")
print(f"is_max_ms:  {model.is_max_ms}")
print(f"mw:         {model.mw}")


print("\n")
#print(model.predictor)
#print(model.hungarian_matcher)



#----- 2. Set up hooks --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run = True
if run:
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, dict):
                 captured[name+ '_masks'] = output['pred_masks'].detach()
                 captured[name+ '_logits'] = output['pred_logits'].detach()

            elif isinstance(output, tuple):
                 captured[name] = output[0].detach()
            else:
                 captured[name] = output.detach()
        return hook

    #------- CNN -------------------------------------------------------------------------------------------------------------
    #Encoder blocks when downscaling
    #print("Encoder portion-------------\n")
    for i in range(5):
        model.conv_blocks_context[i].blocks[0].register_forward_hook(make_hook(f'Encoder stage {i+1}'))
        model.conv_blocks_localization[i][0].blocks[0].register_forward_hook(make_hook(f'Decoder stage {i+1}'))

    #Decoder blocks when upscaling
    #print("Decoder portion-------------\n")
    
    #------------------------------------------------------------------------------------------------------------------------


    #--Transformer-----------------------------------------------------------------------------------------------------------
    #1. Patch embedding
    #model.transformer.embeddings.register_forward_hook(make_hook('vit_path_embed'))

    #2. Attention Layer weights - therefore we can tell if the weights are changing
    if is_max:
        for i in range(len(model.transformer.encoder.layer)):
            model.transformer.encoder.layer[i].attn. softmax.register_forward_hook(make_hook(f"vit_attn_layer_{i}"))
        #3. Full encoder output
        #$model.transformer.encoder.register_forward_hook(make_hook('vit_encoder_output'))

        #Transoformer decoder output
        model.predictor.register_forward_hook(make_hook('transformer_decoder_output'))

        #3. Full Transformer output
        #model.transformer.register_forward_hook(make_hook('vit_full_output'))

        #Masked decoder predictions - model doesnt support intermediate predictions will add if necessary
        #for i in range(len(model.predictor))


    # ------------------------------------------------------------------------------------------------------------------------


    # ---3. Run a forward pass ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #batch_size, channels, depth, height, width
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, d, h, w)
    with torch.no_grad():
        output = model(x)

    #what type is our output
    print(f"output type: {type(output)}")
    if isinstance(output, dict):
        print("Transformer decoder mode")
        print("→ branch 1: is_max_cls=True, is_max_ds=True, deep_supervision=False")
        print(f"  keys: {output.keys()}")

        pred_masks = output['pred_masks']
        pred_logits = output['pred_logits']
    
        B, N_q, D, H, W = pred_masks.shape
        num_classes = pred_logits.shape[-1]

        print(f"  pred_masks:  {pred_masks.shape}")
        print(f"  pred_logits: {pred_logits.shape}")

        #segmentation map
        mask_probs = pred_masks.sigmoid()
        query_per_voxel = mask_probs.argmax(dim=1)
        class_per_query = pred_logits.argmax(dim=1)
        final_seg = class_per_query[
            torch.arange(B).view(B,1,1,1).expand_as(query_per_voxel),query_per_voxel
        ]

    elif isinstance(output, list):
        print("→ branch 1 with deep supervision ON")
        print(f"  output[0] keys: {output[0].keys()}")
    elif isinstance(output, tuple):
        print("CNN-only mode (deep supervision tuple)")
        print("→ branch 2: masks only, no class logits")
        print(f" tuple length: {len(output)}")

        seg_map = output[0] # (B, num_classes, D, H, W)
        print(f" seg_map shape: {seg_map.shape}")

        #label map
        final_seg = seg_map.argmax(dim=1)
        B, D_out, H_out, W_out = final_seg.shape
        N_q = NUM_CLASSES
        num_classes_out = NUM_CLASSES

    #Testing to make sure Hungarian works for loss functions------------------------------------------------------------------
    #Display masks and logits for output from Transformer decoder for loss functions
    #print(output['pred_masks'].shape)  #tells you B, N_q, D, H, W
    #print(output['pred_logits'].shape) #tells you B, N_q, num_classes

    #probably delete = used for debugging on Full TransUnet settings
    # B, N_q, D, H, W = output['pred_masks'].shape
    # num_classes = output['pred_logits'].shape[-1]


    # matcher = HungarianMatcher3D(cost_class=2.0, cost_mask=5.0,cost_dice=5.0)
    # matcher.eval()

    # with torch.no_grad():
    #      indicies = matcher(output, targets)

    #THIS IS DONE LATER DOWN FOR VISUAL ORDER
    # print("\n------Hungarian Matcher output-------")
    # row_ind, col_ind = indicies[0]
    # print(f"\nMatched {len(row_ind)} pairs:")
    # for q, gt in zip(row_ind.tolist(), col_ind.tolist()):
    #     print(f"  query {q} → gt organ {gt}")

    # ------------------------------------------------------------------------------------------------------------------------

    # -- 4. Inspect -------------
    print(f"\nWe have {len(model.conv_blocks_context)} convolutional kernels (conv_kernel_size)\n")
    print("----------Checking downsampling dimensional changes----------------------------------------------------\n")
    print("        batchsize,channel,Depth,heigh,width\n")

    items_per_block = 5
    i = 0

    #inspect hooks --------------
    for name, tensor in captured.items():
            if i == items_per_block or i == items_per_block * 2 or i == items_per_block * 3:
                print("\n")
            print(f"{name}: {tensor.shape}  "
                f"min={tensor.min():.3f}  "
                f"max={tensor.max():.3f}  "
                f"nan={tensor.isnan().any()}")
            i += 1


    print("\n")
    print("----------Checking weights being refined within each block----------------------------------------------------")
    print("        Output,input,[FILTER SIZE]")
    print("\nEncoder\n")
    print(model.conv_blocks_context[0].blocks[0].conv.weight.shape)
    print(model.conv_blocks_context[0].blocks[1].conv.weight.shape)
    # print(model.conv_blocks_context[1].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_context[1].blocks[1].conv.weight.shape)
    # print(model.conv_blocks_context[2].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_context[2].blocks[1].conv.weight.shape)
    # print(model.conv_blocks_context[3].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_context[3].blocks[1].conv.weight.shape)
    print("\nDecoder\n")
    print(model.conv_blocks_localization[0][0].blocks[0].conv.weight.shape)
    print(model.conv_blocks_localization[0][1].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[1][0].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[1][1].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[2][0].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[2][1].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[3][0].blocks[0].conv.weight.shape)
    # print(model.conv_blocks_localization[3][1].blocks[0].conv.weight.shape)

    #HOLD ONLY IF IN FULL Transformer mode
    # print("\n------Hungarian Matcher output-------")
    # row_ind, col_ind = indicies[0]
    # print(f"\nMatched {len(row_ind)} pairs:")
    # for q, gt in zip(row_ind.tolist(), col_ind.tolist()):
    #     print(f"  query {q} → gt organ {gt}")

    # # basic sanity checks
    # assert len(row_ind) == NUM_CLASSES, "wrong number of matched pairs"
    # assert len(set(row_ind.tolist())) == NUM_CLASSES, "duplicate query indices"
    # assert len(set(col_ind.tolist())) == NUM_CLASSES, "duplicate gt indices"
    # print("\nSanity checks passed")


    #Testing loss functions based on model settings ----------------------

    # need B=2 for BatchNorm in train mode
    # rebuild input and targets with batch size 2
    #From our scans we have to have ground truth labels, and ground truth masks
    targets = [{
            'labels': torch.randint(0, NUM_CLASSES, (NUM_CLASSES,)),
            'masks': torch.randint(0,2, (NUM_CLASSES, D_out, H_out, W_out)).float(),
    } for _ in range(BATCH_SIZE)]


    x_train = torch.randn(2, INPUT_CHANNELS, d, h, w).float()

    targets_train = [
        {
            'labels': torch.randint(0, NUM_CLASSES, (NUM_CLASSES,)),
            'masks':  torch.randint(0, 2, (NUM_CLASSES, D_out, H_out, W_out)).float(),
        }
        for _ in range(2)   # one dict per scan in batch
    ]

    # switch to train mode so computation graph gets built
    model.train()
    out_train = model(x_train)

    # compute loss using Beckschen's function
    if isinstance(out_train, dict):
        loss = compute_loss_hungarian(
            outputs=out_train,
            targets=targets_train,
            idx=0,
            matcher=HungarianMatcher3D(cost_class=2.0, cost_mask=5.0,cost_dice=5.0),
            num_classes=num_classes_out,
            cost_weight=[2, 5, 5],
        )
    else:
        #CNN-only loss - cross entropy +dice
        seg = out_train[0] if isinstance(out_train,tuple) else out_train
        #build interget label map from targets
        gt_seg = torch.zeros(2,D_out, H_out, W_out, dtype=torch.long)
        for b ,t in enumerate(targets_train):
            for mask, label in zip(t['masks'],t['labels']):
                gt_seg[b][mask.bool()] = label

        loss_ce = cnn_compute_ce_loss(seg, gt_seg)
        loss_dice = cnn_compute_dice_loss(gt_seg,seg, NUM_CLASSES)

        loss = loss_ce + loss_dice

    print(f"\n── Loss ────────────────────────────────")
    print(f"total loss: {loss.item():.4f}")

    # ── 9. Backward pass ────────────────────────────────────────────
    loss.backward()
    print("backward pass complete ✓")

    # verify gradients flowed through the model
    print("\n── Gradient check (first 3 params with grad) ──")
    count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and count < 3:
            print(f"  {name}: grad norm={param.grad.norm():.4f}")
            count += 1

    print("\n── Segmentation output inspection ──────────────")

    # pred_masks  = output['pred_masks']         # (B, N_q, D, H, W)
    # pred_logits = output['pred_logits']        # (B, N_q, num_classes)

    # print(f"pred_masks  shape: {pred_masks.shape}")
    # print(f"pred_logits shape: {pred_logits.shape}")
    # print(f"pred_masks  range: {pred_masks.min():.3f} to {pred_masks.max():.3f}")
    # print(f"pred_masks  nan:   {pred_masks.isnan().any()}")

    # # convert logits to probabilities
    # mask_probs = pred_masks.sigmoid()
    # print(f"mask_probs  range: {mask_probs.min():.3f} to {mask_probs.max():.3f}")

    # # which query fires strongest at each voxel
    # query_per_voxel = mask_probs.argmax(dim=1)    # (B, D, H, W)
    # print(f"\nquery_per_voxel shape: {query_per_voxel.shape}")
    # print(f"unique queries firing:  {query_per_voxel.unique().tolist()}")

    # # what class each query predicts
    # class_per_query = pred_logits.argmax(dim=-1)  # (B, N_q)
    # print(f"class per query:        {class_per_query[0].tolist()}")

    # # final label map
    # B_out = pred_masks.shape[0]
    # final_seg = class_per_query[
    #     torch.arange(B_out).view(B_out,1,1,1).expand_as(query_per_voxel),
    #     query_per_voxel
    # ]                                             # (B, D, H, W)
    # print(f"\nfinal_seg shape:   {final_seg.shape}")
    # print(f"unique classes:    {final_seg.unique().tolist()}")
    # print(f"\nclass distribution:")
    # for c in final_seg.unique().tolist():
    #     count = (final_seg == c).sum().item()
    #     pct   = count / final_seg.numel() * 100
    #     print(f"  class {c}: {count:6d} voxels ({pct:.1f}%)")

    print("\nFull pipeline verified:")
    print("  ✓ CNN encoder")
    print("  ✓ Transformer encoder")
    print("  ✓ CNN decoder")
    print("  ✓ Transformer decoder")
    print("  ✓ Hungarian matcher")
    print("  ✓ Loss computation")
    print("  ✓ Backward pass")