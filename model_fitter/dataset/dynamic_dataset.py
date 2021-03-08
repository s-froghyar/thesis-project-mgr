import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split

from .dataset_utils import BASE_SAMPLE_RATE
from .transformations import apply_audio_transformations

torchaudio.set_audio_backend("sox_io")

class GtzanDynamicDataset(Dataset):
    '''
        - X: wave_data
        - y: label
        - dataset_params:
            - bands
            - window_size
            - hop_size
    '''
    def __init__(self, X, y, dataset_params, device, train=False, augerino=False):
        self.X = X
        self.targets = y
        self.device = device
        self.train = train

        if augerino:
            self.augerino = augerino
        else:
            self.dataset_params = dataset_params
            self.e0 = dataset_params["e0"]
            
            self.mel_spec_transform = aud_transforms.MelSpectrogram(
                sample_rate=BASE_SAMPLE_RATE,
                n_mels=dataset_params["bands"],
                n_fft=dataset_params["window_size"],
                hop_length=dataset_params["hop_size"]
            )

        
    def __getitem__(self, index):
        if self.augerino:
            return self.get_wave_output(index)
        else:
            return self.get_transformed_output(index)
         
    def __len__(self):
        return len(self.X)
    def transform(self, x):
        x = torch.from_numpy(x).to(torch.float32)
        return self.mel_spec_transform(x)

    def get_wave_output(self, index):
        return (
            self.X[index],
            self.X[index],
            self.targets[index]
        )
    def get_transformed_output(self, index):
        transformed_spectrogram = self.transform(
            apply_audio_transformations(self.X[index], self.e0)
        )
        return (
            self.transform(self.X[index]),
            transformed_spectrogram,
            self.targets[index]
        )