import torch

def unif_aug_loss(augs, aug_reg=0.01):
    aug_loss = 0
    # if both transformations are applied
    if len(augs) == 4:
        lims_noise =  augs[0].lims
        lims_pitch =  augs[1].lims
        aug_loss -= torch.abs(lims_noise[0] - lims_noise[1]) * aug_reg
        aug_loss -= torch.abs(lims_pitch[0] - lims_pitch[1]) * aug_reg
    else:
        lims_aug =  augs[0].lims
        aug_loss -= torch.abs(lims_aug[0] - lims_aug[1]) * aug_reg
    
    return aug_loss
