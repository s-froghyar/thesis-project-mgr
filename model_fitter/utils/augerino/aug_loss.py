import torch

def unif_aug_loss(augs, aug_reg=0.01):
    aug_loss = 0
    
    lims_aug =  augs.lims
    aug_loss -= torch.abs(lims_aug[0] - lims_aug[1]) * aug_reg
    
    return aug_loss
