import torch

def unif_aug_loss(outputs, labels, model,
                  base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01):

    base_loss = base_loss_fn(outputs, labels)
    
    sp = torch.nn.Softplus()
    width = sp(model.aug.width)
    aug_loss = (width).norm()
    
    return base_loss - reg * aug_loss
