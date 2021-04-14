import torch
import torch.nn as nn
import numpy as np
import gc

from .augerino import *
from .tp import *


def get_final_preds(all_preds, device):
    bs = all_preds[0].shape[0]
    out = torch.from_numpy(np.zeros((bs, 10))).to(dtype=torch.float32, device=device)
    for pred in all_preds:
        out += pred
    return torch.div(out, 4)

def generate_batch_of_spectrograms(data, config, device):
    batch_specs = torch.from_numpy(np.zeros((data.shape[0], 6, 256, 76)))
    for b in range(len(data)):
        specs = get_6_spectrograms(data[b], config, device)
        batch_specs[b] = specs
    batch_specs.requires_grad_(True)
    return batch_specs.to(dtype=torch.float32, device=device)

def get_model_prediction(model, batch_specs, device, model_type):
    ''' Gets the sum of the predictions for the 12 patches of the spectrogram '''
    if model_type == 'augerino':
        return model(batch_specs)
    num_of_patches = batch_specs.size(1)
    preds_sum = torch.from_numpy(np.zeros((batch_specs.shape[0], 10))).to(dtype=torch.float32, device=device)
    for i in range(num_of_patches):
        strip_data = batch_specs[:,i,:,:].float()
        preds = model(strip_data)
        preds_sum += preds.to(dtype=torch.float32, device=device)
    return preds_sum


def get_model_loss(model, predictions, targets, config, device, x=None, transformed_data=None):
    ''' Gets the losses for the model. Returns tuple of (model_loss, tp_loss, augerino_loss) '''
    targets = targets.to(device)
    num_of_patches = x.size(1)
    if len(transformed_data) > 0:
        transformed_data = transformed_data.to(device)
    base_loss = config.loss(predictions, targets)
    tp_loss = 0.0
    augerino_loss = 0.0

    if config.is_tangent_prop:
        for i in range(num_of_patches):
            tp_loss += config.gamma * get_tp_loss(x[:,i,:,:], predictions, config.e0, device, transformed_data[:,i,:,:], model)
            gc.collect()
            torch.cuda.empty_cache()
        tp_loss = tp_loss / num_of_patches
        base_loss = base_loss + tp_loss
    elif config.augerino:
        augerino_loss = unif_aug_loss(model.aug)
        base_loss = base_loss + augerino_loss

    return base_loss, tp_loss, augerino_loss

def init_layer(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        layer.bias.data.fill_(0.01)

def get_6_spectrograms(wd, config, device):
    patches = np.array_split(wd, 6)
    out = torch.from_numpy(np.zeros((6, 256, 76)))
    for ind, patch in enumerate(patches):
        patch = patch.to(dtype=torch.float32, device=device)
        out[ind] = config.mel_spec_transform(patch)
    return out
def get_batch_data(base, model_type, aug_ind):
    if model_type == 'segmented':
        return base[:,aug_ind, :, :, :]
    elif model_type == 'augerino':
        return base
    elif model_type == 'tp':
        return base