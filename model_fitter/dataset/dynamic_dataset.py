import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split

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
    def __init__(self, X, y, dataset_params, e0, device, train=False):
        self.X = X
        self.targets = y

        self.device = device
        self.dataset_params = dataset_params
        self.e0 = e0
        
        self.mel_spec_transform = aud_transforms.MelSpectrogram(
            sample_rate=BASE_SAMPLE_RATE,
            n_mels=dataset_params["bands"],
            n_fft=dataset_params["window_size"],
            hop_length=dataset_params["hop_size"]
        )
        self.train = train
    def __getitem__(self, index):
        transformed_spectrogram = self.transform(
            apply_audio_transformations(self.X[index])
        )
        return (
            self.transform(self.X[index]),
            transformed_spectrogram,
            self.targets[index]
        ) 
    def __len__(self):
        return len(self.X)
    def transform(self, x):
        x = torch.from_numpy(x).to(self.device)
        return self.mel_spec_transform(x)