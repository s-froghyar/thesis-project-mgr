from torch import nn
import numpy as np
import torch
import torchaudio.transforms as aud_transforms

from .utils import *
from .tp import get_tp_loss
from .augerino import unif_aug_loss, GaussianNoiseAug, PitchShiftAug

def train_model(model, config, reporter, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (base_data, transformed_data, targets) in enumerate(loader):
        if config.is_tangent_prop:
            base_data.requires_grad = True
        targets = targets.to(device)

        n_augs = 1
        
        if config.model_type == 'segmented':
            n_augs = len(config.aug_params.get_options_of_chosen_transform())

        for i in range(n_augs):

            data = get_batch_data(base_data, config.model_type, i).to(device)
            predictions = get_model_prediction(model, data, device, config.model_type)

            loss, tp_loss, augerino_loss = get_model_loss(  model,
                                                            predictions,
                                                            targets,
                                                            config,
                                                            device,
                                                            x=data,
                                                            transformed_data=transformed_data)
            reporter.record_batch_data(predictions, targets, (loss, tp_loss, augerino_loss))
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
    if config.model_type == 'augerino':
        print(f"limits: {model.aug[0].lims}")

