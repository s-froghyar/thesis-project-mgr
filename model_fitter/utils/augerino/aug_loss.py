import torch

def unif_aug_loss(augs, aug_reg=0.01):
    sp = torch.nn.Softplus()
    lims_aug =  sp(augs.log_lims)
    aug_loss = (lims_aug).norm()
    
    return aug_loss
