import os
import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as aud_transforms
from sklearn.model_selection import train_test_split

from .utils import *
import pickle
torchaudio.set_audio_backend("sox_io")

df = None
BASE_SAMPLE_RATE = 16000
genre_mapping = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9, 
}

class GtzanWave:
    """
    GTZAN Data generator using data augmentation techniques
    Only wave data is the output
    Options (for now):
        - noise-injection: (0, 0.2, step=0.001)
        - pitch-shift: (-5, 5, 0.5)
    """
    def __init__(
        self,
        aug_params,
        test_size=0.1,
    ):
        self.aug_params = aug_params
        init_x, self.test_x, init_y, self.test_y = train_test_split(df['filePath'],
                                                                df['label'],
                                                                test_size=test_size)
        self.prep_test_values()
        self.init_dataframe(init_x, init_y)
        self.give_report()
    
    def __len__(self):
        return len(self.train_y)
    def __getitem__(self, index):
        return self.train_x[index]

    def prep_test_values(self):
        ''' generate the spectrograms for the test values and store them'''

        print("Preparing test values...")
        new_test_x = []
        new_test_y = []
        for index, path in self.test_x.iteritems():
            wave_data, sample_rate = librosa.core.load(path, 
                                               sr    = BASE_SAMPLE_RATE,
                                               mono  = True,
                                               dtype = np.float32)
            
            new_test_x.append(get_correct_input_format(wave_data, self.aug_params.segmented))
            new_test_y.append(genre_mapping[str(self.test_y[index])])
            
        self.test_x = np.array(new_test_x)
        self.test_y = np.array(new_test_y)

    def init_dataframe(self, init_x, init_y):
        noise_injection = self.aug_params.noise_injection
        pitch_shift = self.aug_params.pitch_shift
        
        self.set_up_buckets(init_x, init_y)

        print("Data augmentation started...")
        
        NOISE_INJECTION_STEPS = ((noise_injection[1] - noise_injection[0]) / noise_injection[2])
        PITCH_SHIFT_STEPS = ((pitch_shift[1] - pitch_shift[0]) / pitch_shift[2])
        NUM_OF_AUGMENTED_DATA = (len(self.train_x)) * (NOISE_INJECTION_STEPS + PITCH_SHIFT_STEPS)
        
        for index, filePath in init_x.iteritems():
            wave_data, sample_rate = librosa.core.load(filePath, 
                                            sr    = BASE_SAMPLE_RATE,
                                            mono  = True,
                                            dtype = np.float32)
            self.create_noise_injected_data(wave_data,
                                        sample_rate,
                                        noise_injection,
                                        genre_mapping[str(init_y[index])])
            self.create_pitch_shifted_data(wave_data,
                                        sample_rate,
                                        pitch_shift,
                                        genre_mapping[str(init_y[index])])

        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y) 
    
    def create_noise_injected_data(self, wd, sr, ni, label):
        for noise_factor in np.arange(ni[0] + ni[2], ni[1] + ni[2], ni[2]):
            if noise_factor == 0: continue

            noise = np.random.randn(len(wd))
            augmented_data = wd + noise_factor * noise

            self.train_x.append(get_correct_input_format(augmented_data, self.aug_params.segmented))
            self.train_y.append(label)
    
    def create_pitch_shifted_data(self, wd, sr, ps, label):
        for pitch_factor in np.arange(ps[0], ps[1], ps[2]):
            if pitch_factor == 1: continue
            
            augmented_data = librosa.effects.pitch_shift(wd, sr, pitch_factor)
            
            self.train_x.append(get_correct_input_format(augmented_data, self.aug_params.segmented))
            self.train_y.append(label)
            
    def set_up_buckets(self, init_x, init_y):
        ''' initialising  '''

        print("Preparing original train data...")

        self.train_x = []
        self.train_y = []

        for index, row in init_x.iteritems():
            wave_data, sample_rate = librosa.core.load(row, 
                                            sr    = BASE_SAMPLE_RATE,
                                            mono  = True,
                                            dtype = np.float32)
            self.train_x.append(get_correct_input_format(wave_data, self.aug_params.segmented))
            self.train_y.append(genre_mapping[str(init_y[index])])         
            
    def give_report(self):
        print("Data Augmentation is completed with results:")
        print(f"Training samples: {len(self.train_x)}")
        print(f"Testing samples: {len(self.test_x)}")
        print(f"Each input is {self.all_shapes_same(self.train_x[0].shape)}")
    
    def all_shapes_same(self, base_shape):
        shapes = []
        for sample in self.train_x:
            if sample.shape != base_shape:

                return f"not the same shape as base_shape of {base_shape}"
        return f"the same shape, that is {base_shape}"

class GtzanDataset(Dataset):
    def __init__(self, X, y, dataset_params, train=False):
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

def load_data(data_path, aug_params, is_local=True):
    print("Loading in data...")
    test_file_path = f"{data_path}/gtzan_test"
    test_file_exists = os.path.isfile(test_file_path)
    global df
    if is_local:
        if test_file_exists:
            with open(test_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            df = get_data_frame(data_path, True)
            temp = GtzanWave(aug_params)
            with open(test_file_path, 'wb') as f:
                pickle.dump(temp, f)
            return temp
    else:
        df = get_data_frame(data_path, False)
        return GtzanWave(aug_params)

        
        

