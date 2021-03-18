import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

from .dataset_utils import BASE_SAMPLE_RATE, generate_6_strips
from .transformations import gaussian_noise_injection, pitch_shift

torchaudio.set_audio_backend("sox_io")

class GtzanDynamicDataset(Dataset):
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
        model_type='segmented'
    ):
        self.X = paths
        self.targets = labels
        self.aug_params = aug_params
        self.device = device
        self.train = train

        self.augmentations = {
            'ni': gaussian_noise_injection,
            'ps': pitch_shift
        }
        self.model_type = model_type

        if model_type != 'augerino':
            self.e0 = mel_spec_params["e0"]
            
            self.mel_spec_transform = aud_transforms.MelSpectrogram(
                sample_rate=BASE_SAMPLE_RATE,
                n_mels=mel_spec_params["bands"],
                n_fft=mel_spec_params["window_size"],
                hop_length=mel_spec_params["hop_size"]
            ).to(self.device)

        
    def __getitem__(self, index):
        path = self.X[index]
        wave_data, sample_rate = librosa.core.load(path, 
                                               sr    = BASE_SAMPLE_RATE,
                                               mono  = True,
                                               dtype = np.float32)
        
        if self.model_type == 'augerino':
            return (torch.tensor(np.array_split(wave_data[:465984], 6)), [], [], self.targets[index])
        
        aug_type = self.aug_params.transform_chosen
        
        if self.model_type == 'tp':
            return  (
                self.get_6_spectrograms(wave_data),
                self.get_6_spectrograms(self.augmentations[aug_type](wave_data, self.e0)),
                [],
                self.targets[index]
            )
        else:
            aug_options = self.aug_params.get_options_of_chosen_transform()
            augs = []
            for aug_factor in aug_options:
                augs.append(
                    self.get_6_spectrograms(self.augmentations[aug_type](wave_data, aug_factor))
                )
            return (
                self.get_6_spectrograms(wave_data),
                [],
                augs,
                self.targets[index]
            )

         
    def __len__(self):
        return len(self.X)
    def transform(self, x):
        x = torch.from_numpy(x).to(torch.float32)
        return self.mel_spec_transform(x)

    def get_6_spectrograms(self, wd):
        patches = np.array_split(wd[:465984], 6)
        out = torch.from_numpy(np.zeros((6, 256, 76)))
        for ind, patch in enumerate(patches):
            patch = torch.from_numpy(patch).to(dtype=torch.float32, device=self.device)
            out[ind] = self.mel_spec_transform(patch)
        return out.to(dtype=torch.float32, device=self.device)