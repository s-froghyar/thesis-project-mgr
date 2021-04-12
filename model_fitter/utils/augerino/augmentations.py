import torch
import torch.nn as nn
import librosa
import numpy as np
from numpy.random import default_rng

BASE_SAMPLE_RATE = 16000


class GaussianNoiseAug(nn.Module):
    """
    Differetiable Gaussian noise injection
    """
    def __init__(self):
        super().__init__()
        self.aug=True
        self.log_lims = nn.Parameter(torch.tensor([0., 1.]))

    @property
    def lims(self):
        return torch.sigmoid(self.log_lims) * 2 - 1

    def forward(self, x):
        bs = x.shape[0]
        rng = default_rng()
        snr_range = torch.tensor([self.lims[0] * 12, self.lims[1] * 12])
        snr = torch.rand(bs, device=self.lims.device) * (snr_range[1] - snr_range[0]) + snr_range[0]

        RMS_s = torch.sqrt(torch.mean(x**2, 1))
        RMS_n = torch.sqrt(RMS_s**2/(pow(10,snr/10)))

        noise = torch.zeros_like(x)
        for i in range(bs):
            noise[i,:] = torch.FloatTensor(x.shape[1]).normal_(0, RMS_n[i])
        
        return torch.add(x, noise)

class PitchShiftAug(nn.Module):
    """
    Differetiable pitch shift
    """
    def __init__(self):
        super().__init__()
        self.aug=True
        self.log_lims = nn.Parameter(torch.tensor([-1., 1.]))

    @property
    def lims(self):
        return torch.sigmoid(self.log_lims) * 2 - 1

    def forward(self, x):
        bs = x.shape[0]
        factor_range = torch.tensor([self.lims[0] * 3, self.lims[1] * 3])
        out = torch.zeros_like(x)
        factor = torch.rand(bs, device=self.lims.device) * (factor_range[1] - factor_range[0]) + factor_range[0]
        for index, row in enumerate(x):
            out[index] = torch.from_numpy(librosa.effects.pitch_shift(row.numpy(), BASE_SAMPLE_RATE, factor[index]))
        return out.to(self.lims.device)
