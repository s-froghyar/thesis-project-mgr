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
        #RMS values of noise
        RMS_n=torch.sqrt(RMS_s**2/(pow(10,snr/10)))
        #Additive white gausian noise. Thereore mean=0
        #Because sample length is large (typically > 40000)
        #we can use the population formula for standard daviation.
        #because mean=0 STD=RMS
        STD_n=RMS_n
        noise = torch.zeros_like(x)
        for i in range(bs):
            noise[i,:] = torch.FloatTensor(x.shape[1]).normal_(0, STD_n[i])



        # sample_length = x.shape[1]
        # snr_range = torch.tensor([self.lims[0] * 3, self.lims[1] * 3])
        
        # g_noise = np.random.randn(sample_length)
        # snr = torch.rand(bs, device=self.lims.device) * (snr_range[1] - snr_range[0]) + snr_range[0]

        # noise_power = np.mean(np.power(g_noise, 2))
        # sig_power = torch.mean(torch.pow(x, 2))

        # snr_linear = 10**(snr / 10.0)
        # noise_factor = torch.sqrt( (sig_power / noise_power) * (1 / snr_linear) )

        # # noise_factor = noise_factor.unsqueeze(dim=1).expand(bs, sample_length)
        # g_noise = torch.from_numpy(g_noise)
        # vals = noise_factor * g_noise
        
        return torch.add(x, noise)

class PitchShiftAug(nn.Module):
    """
    Differetiable pitch shift
    """
    def __init__(self):
        super().__init__()
        self.aug=True
        self.log_lims = nn.Parameter(torch.tensor([-12., 12.]))

    @property
    def lims(self):
        return torch.sigmoid(self.log_lims) * 2 - 1

    def forward(self, x):
        bs = x.shape[0]
        out = np.zeros(x.shape)
        factor = torch.rand(bs, device=self.lims.device) * (self.lims[1] - self.lims[0]) + self.lims[0]
        for index, row in enumerate(x):
            out[index] = librosa.effects.pitch_shift(row.detach().numpy(), BASE_SAMPLE_RATE, (1 + factor[index]))
        return torch.from_numpy(out).to(self.lims.device)


def speedx(signal, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = torch.round( torch.range(0, len(signal), factor) )
    indices = indices[indices < len(signal)]
    return sound_array[ indices ]
