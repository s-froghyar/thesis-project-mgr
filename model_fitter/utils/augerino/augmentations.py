import torch
import torch.nn as nn
import librosa
import numpy as np

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
        sample_length = x.shape[1]

        g_noise = np.random.randn(sample_length)
        snr = torch.rand(bs, device=self.lims.device) * (self.lims[1] - self.lims[0]) + self.lims[0]

        noise_power = np.mean(np.power(g_noise, 2))
        sig_power = torch.mean(torch.pow(x, 2))

        snr_linear = 10**(snr / 10.0)
        noise_factor = torch.sqrt( (sig_power / noise_power) * (1 / snr_linear) )

        noise_factor = noise_factor.unsqueeze(dim=1).expand(bs, sample_length)
        g_noise = torch.from_numpy(g_noise)
        vals = noise_factor * g_noise
        
        return torch.add(x, vals)

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
