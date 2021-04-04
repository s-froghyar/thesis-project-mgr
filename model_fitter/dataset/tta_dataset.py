import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

from .dataset_utils import BASE_SAMPLE_RATE, generate_6_strips
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
        model_type='segmented',
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
            'ps': pitch_shift
        }
        self.model_type = model_type

        if model_type != 'augerino':
            self.e0 = mel_spec_params["e0"]
            
            self.mel_spec_transform = nn.Sequential(
                aud_transforms.MelSpectrogram(
                        sample_rate=BASE_SAMPLE_RATE,
                        n_mels=mel_spec_params["bands"],
                        n_fft=mel_spec_params["window_size"],
                        hop_length=mel_spec_params["hop_size"]
                ),
                aud_transforms.AmplitudeToDB()
            )
            


        
    def __getitem__(self, index):
        path = self.X[index]
        wave_data = self.load_audio(path)

        aug_type = self.aug_params.transform_chosen
        aug_options = torch.rand(4, device=self.device) * (self.tta[1] - self.tta[0]) + self.tta[0]

        augs = []
        for aug_factor in aug_options:
            augmented_wd = self.augmentations[aug_type](wave_data, aug_factor)
            augs.append(
                self.get_6_spectrograms(augmented_wd)
            )
        return (torch.stack(augs), self.targets[index])

         
    def __len__(self):
        return len(self.X)

    def get_6_spectrograms(self, wd):
        ''' Transforms wave data to Melspectrogram and returns 6 (256x76) shaped patches '''
        if isinstance(wd, np.ndarray):
            wd = torch.from_numpy(wd[:465984])
        else:
            wd = wd[:465984]
        mel_spec = self.mel_spec_transform(wd).to(self.device) # 456 width
        patches = torch.split(mel_spec, 76, dim=1)
        return torch.stack(patches)

    def load_audio(self, path):
        ''' Loads wave data from given path and resamples it to 16000Hz '''
        wd, sr = torchaudio.load(path, normalize=True)
        resampler = aud_transforms.Resample(sr, BASE_SAMPLE_RATE)
        return resampler(wd).squeeze()
    def transform(self, x):
        return self.mel_spec_transform(x)