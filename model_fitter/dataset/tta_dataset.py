import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

from .dataset_utils import BASE_SAMPLE_RATE, splitsongs
from .transformations import gaussian_noise_injection, pitch_shift

torchaudio.set_audio_backend("sox_io")

class GtzanTTADataset(Dataset):
    '''
    Arguments
        - X: path_to_wave_data
        - y: label
        - dataset_params:
            - bands
            - window_size
            - hop_size
    Returns
        - For augerino adictionary with:
            - "wave_data"   - loaded in from filepath
            - "label"       - label corresponding to the wave data
    '''
    def __init__(
        self,
        paths=None,
        labels=None,
        mel_spec_params=None,
        aug_params=None,
        device=None,
        train=False,
        tta_settings=None
    ):
        self.X = paths
        self.targets = labels
        self.aug_params = aug_params
        self.device = device
        self.train = train
        self.tta = tta_settings
        self.augmentations = {
            'ni': gaussian_noise_injection,
            'ps': pitch_shift,
            'none': lambda x,y,z: x
        }

        self.e0 = mel_spec_params["e0"]
        
        self.mel_spec_transform = nn.Sequential( 
                    aud_transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_mels=128,
                    n_fft=1024,
                    hop_length=256
                )
            ).double().to(self.device)

        
    def __getitem__(self, index):
        path = self.X[index]
        wave_data = self.load_audio(path)

        aug_type = self.aug_params.transform_chosen
        aug_options = self.get_aug_options()

        augs = []
        for aug_factor in aug_options:
            augmented_wd = self.augmentations[aug_type](wave_data, aug_factor, False)
            augs.append(
                self.get_patched_spectrograms(augmented_wd)
            )
        return (torch.stack(augs), self.targets[index])

         
    def __len__(self):
        return len(self.X)
    def set_tta_params(self, params):
        self.tta = params

    def get_patched_spectrograms(self, wd):
        ''' Transforms wave data to Melspectrogram and returns 6 (256x76) shaped patches '''
        if isinstance(wd, np.ndarray):
            wd = torch.from_numpy(wd[:478912])
        else:
            wd = wd[:478912]

        patches = splitsongs(wd)
        mel_specs = []
        for patch in patches:
            patch = patch.to(self.device)
            try:
                mel_specs.append(self.mel_spec_transform(patch))
            except:
                print('patch device', patch.device)
                raise ValueError('something with the device again')

        return torch.stack(mel_specs)

    def load_audio(self, path):
        ''' Loads wave data from given path and resamples it to 16000Hz '''
        wd, sr = torchaudio.load(path, normalize=True)
        resampler = aud_transforms.Resample(sr, BASE_SAMPLE_RATE)
        return resampler(wd).squeeze()
    def transform(self, x):
        return self.mel_spec_transform(x)
    def get_aug_options(self):
        if self.tta is None:
            return [0]
        return torch.rand(4, device=self.device) * (self.tta[1] - self.tta[0]) + self.tta[0]
