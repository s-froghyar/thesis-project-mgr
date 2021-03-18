from torch import nn
import numpy as np
import torch

from .tp import get_tp_loss
from .augerino import unif_aug_loss

def train_model(model, config, reporter, device, loader, optimizer, epoch):
    model.train()
    reporter.reset_epoch_data()

    for batch_idx, (base_data, transformed_data, augmentations, targets) in enumerate(loader):
        print(base_data.shape)
        if config.is_tangent_prop:
            base_data.requires_grad = True

        n_augs = 1
        if config.model_type == 'segmented': n_augs = len(config.aug_params.get_options_of_chosen_transform()) + 1
                
        for i in range(n_augs):
            predictions = get_model_prediction(model, base_data, targets, device, config)

            loss, tp_loss, augerino_loss = get_model_loss(  model,
                                                            predictions,
                                                            targets,
                                                            config,
                                                            device,
                                                            x=base_data,
                                                            transformed_data=transformed_data)
            reporter.record_batch_data(predictions, targets, loss)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Adam step
            optimizer.step()
        
            if batch_idx % config.log_interval == 0:
                reporter.keep_log(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTP-loss: {:.6f}\tAugerino-loss: {:.6f}'.format(
                    epoch, batch_idx * len(base_data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item(), tp_loss, augerino_loss)
                )
    # reporter.record_epoch_data(model, epoch)


def test_model(model, device, loader):
    model.eval()
    
def generate_batch_of_spectrograms(data, config, device):
    batch_specs = torch.from_numpy(np.zeros((data.shape[0], 6, 256, 76)))
    for b in range(len(data)):
        specs = get_6_spectrograms(data[b], config, device)
        batch_specs[b] = specs
    batch_specs.requires_grad_(True)
    return batch_specs.to(dtype=torch.float32, device=device)

def get_model_prediction(model, batch_specs, targets, device, config):
    preds_sum = torch.from_numpy(np.zeros((batch_specs.shape[0], 10))).to(dtype=torch.float32, device=device)
    for i in range(6):
        strip_data = None
        if config.model_type == 'augerino':
            strip_data = batch_specs[:,i,:]
        else:
            strip_data = batch_specs[:,i,:,:]

        strip_data.requires_grad_(True)
        # targets = targets.to(dtype=torch.float32, device=device)
        preds = model(strip_data)
        preds_sum += preds.to(dtype=torch.float32, device=device)
    return preds_sum


def get_model_loss(model, predictions, targets, config, device, x=None, transformed_data=None):
    base_loss = config.loss(predictions, targets)
    tp_loss = 0
    augerino_loss = 0

    if config.is_tangent_prop:
        for i in range(6):
            tp_loss += config.gamma * get_tp_loss(x[:,i,:,:], predictions, config.e0, device, transformed_data[:,i,:,:], model)
    elif config.augerino:
        augerino_loss = unif_aug_loss(model.aug)
    
    model_loss = torch.add(base_loss, tp_loss + augerino_loss)
    return model_loss, tp_loss, augerino_loss

def init_layer(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

def get_6_spectrograms(wd, config, device):
    patches = np.array_split(wd, 6)
    out = torch.from_numpy(np.zeros((6, 256, 76)))
    for ind, patch in enumerate(patches):
        patch = patch.to(dtype=torch.float32, device=device)
        out[ind] = config.mel_spec_transform(patch)
    return out
