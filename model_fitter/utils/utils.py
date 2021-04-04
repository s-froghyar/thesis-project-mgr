import torch
import torch.nn as nn
import numpy as np

from .augerino import *
from .tp import *


def get_final_preds(all_preds):
    bs = all_preds[0].shape[0]
    out = torch.from_numpy(np.zeros((bs, 10))).to(dtype=torch.float32)
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
    ''' Gets the sum of the predictions for the 6 patches of the spectrogram '''
    preds_sum = torch.from_numpy(np.zeros((batch_specs.shape[0], 10))).to(dtype=torch.float32, device=device)
    for i in range(6):
        strip_data = batch_specs[:,i,:,:]
        strip_data.requires_grad_(True)
        preds = model(strip_data)
        preds_sum += preds.to(dtype=torch.float32, device=device)
    return preds_sum


def get_model_loss(model, predictions, targets, config, device, x=None, transformed_data=None):
    ''' Gets the losses for the model. Returns tuple of (model_loss, tp_loss, augerino_loss) '''
    targets = targets.to(device)
    base_loss = config.loss(predictions, targets)
    tp_loss = 0
    augerino_loss = 0

    if config.is_tangent_prop:
        for i in range(6):
            tp_loss += config.gamma * get_tp_loss(x[:,i,:,:], predictions, config.e0, device, transformed_data[:,i,:,:], model)
            tp_loss = tp_loss / 6
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
def get_batch_data(base, model_type, aug_ind):
    if model_type == 'segmented':
        return base[:,aug_ind, :, :, :]
    elif model_type == 'augerino':
        return base
    elif model_type == 'tp':
        return base