
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms

torchaudio.set_audio_backend("sox_io")

class GtzanPreAugmentedDataset(Dataset):
    def __init__(self, X, y, dataset_params, device, train=False):
        self.X = X
        self.targets = y
        self.dataset_params = dataset_params
        self.mel_spec_transform = aud_transforms.MelSpectrogram(
            sample_rate=BASE_SAMPLE_RATE,
            n_mels=dataset_params["bands"], 
            n_fft=dataset_params["window_size"],
            hop_length=dataset_params["hop_size"]
        )
        self.train = train
    def __getitem__(self, index):
        return (self.transform(self.X[index]), self.targets[index]) 
    def __len__(self):
        return len(self.X)
    def transform(self, x):
        x = torch.from_numpy(x).to(torch.float32)
        return self.mel_spec_transform(x)


