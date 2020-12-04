import seaborn as sns
import os
import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
# REAL DATA
DATA_PATH = 'GTZAN'
# df = pd.read_csv('%s/features_30_sec.csv' % DATA_PATH)
# df['filePath'] = DATA_PATH + '/genres_original/' + df['label'] + '/' + df['filename']
# df = getDataFrame(True)
# TEST DATA
# df = pd.read_csv('test.csv')
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

class GtzanData:
    """
    GTZAN Data generator using data augmentation techniques
    Options (for now):
        - noise-injection: (0, 0.2, step=0.001)
        - pitch-shift: (-5, 5, 0.5)
    """
    def __init__(
        self,
        noise_injection=(0.0, 0.1, 0.02),
        pitch_shift=(-5, 0, 1),
        test_size=0.1,
    ):
        init_x, self.test_x, init_y, self.test_y = train_test_split(df['filePath'],
                                                                df['label'],
                                                                test_size=test_size)
        print(len((init_x, init_y)))
        self.prep_test_values()
        self.init_dataframe(init_x, init_y, noise_injection, pitch_shift)
    
    def __len__(self):
        return len(self.train_y)
    def __getitem__(self, index):
        return self.train_x[index]
    def prep_test_values(self):
        new_test_x = []
        new_test_y = []
        for index, path in self.test_x.iteritems():
            new_test_x.append(create_spectrogram_from_filepath(path))
            new_test_y.append(genre_mapping[str(self.test_y[index])])
        self.test_x = np.array(new_test_x)
        self.test_y = np.array(new_test_y)
        print('Tests created: ', self.test_x.shape, self.test_y.shape)
        print(self.test_x[0])
    def init_dataframe(self, init_x, init_y, noise_injection, pitch_shift):
        self.set_up_buckets(init_x, init_y)

        NOISE_INJECTION_STEPS = ((noise_injection[1] - noise_injection[0]) / noise_injection[2])
        PITCH_SHIFT_STEPS = ((pitch_shift[1] - pitch_shift[0]) / pitch_shift[2])
        NUM_OF_AUGMENTED_DATA = (len(self.train_x)) * (NOISE_INJECTION_STEPS + PITCH_SHIFT_STEPS)
        print('Entering init_dataframe loop')
        
        for index, filePath in tqdm(init_x.iteritems()):
            wave_data, sample_rate = librosa.core.load(filePath, 
                                               sr    = None,
                                               mono  = True,
                                               dtype = np.float32)
            # We have the wave data now lets augment it -> Noise injection first
            self.create_noise_injected_data(wave_data,
                                        sample_rate,
                                        noise_injection,
                                        genre_mapping[str(init_y[index])])
            self.create_pitch_shifted_data(wave_data,
                                        sample_rate,
                                        pitch_shift,
                                        genre_mapping[str(init_y[index])])
        # Now that its done lets turn it all to tensors
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        print(self.train_x.shape, self.train_y.shape)
    
    
    def create_noise_injected_data(self, wd, sr, ni, label):
        for noise_factor in np.arange(ni[0] + ni[2], ni[1] + ni[2], ni[2]):
            if noise_factor == 0: continue

            print('noise factor: %f' % (noise_factor))
            noise = np.random.randn(len(wd))
            augmented_data = wd + noise_factor * noise
            self.train_x.append(create_spectrogram(augmented_data.astype(type(wd[0])), sr, 16000))
            self.train_y.append(label)
    def create_pitch_shifted_data(self, wd, sr, ps, label):
        for pitch_factor in np.arange(ps[0], ps[1], ps[2]):
            if pitch_factor == 1: continue

            print('pitch factor: %f' % (pitch_factor))
            augmented_data = librosa.effects.pitch_shift(wd, sr, pitch_factor)
            self.train_x.append(create_spectrogram(augmented_data.astype(type(wd[0])), sr, 16000))
            self.train_y.append(label)
    def set_up_buckets(self, init_x, init_y):
        self.train_x = []
        self.train_y = []
        print('Entering set_up_buckets loop')
        print('num of iterations: %i' % (init_x.size))

        for index, row in tqdm(init_x.iteritems()):
            wave_data, sample_rate = librosa.core.load(row, 
                                               sr    = None,
                                               mono  = True,
                                               dtype = np.float32)
            self.train_x.append(create_spectrogram(wave_data, sample_rate, 16000))
            self.train_y.append(genre_mapping[str(init_y[index])])
        print(len(self.train_x), self.train_y)
        print(self.train_x[0])

class GtzanDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __getitem__(self, index):
        if self.transform is not None:
            return (self.transform(self.X[index]), self.y[index]) 
        return (self.X[index], y[index])
    def __len__(self):
        return len(self.X)  




def create_spectrogram_from_filepath(filePath):
    wave_data, sample_rate = librosa.core.load(filePath, 
                                               sr    = None,
                                               mono  = True,
                                               dtype = np.float32)
    return create_spectrogram(wave_data, sample_rate)
def create_spectrogram(wave_data, sample_rate, sample_rate_new = 16000):
    # downsample to fs = 16kHz 
    wave_data = librosa.core.resample(wave_data, 
                          sample_rate, 
                          sample_rate_new)

    sample_rate = sample_rate_new

    # normalize input amplitude
    wave_data = librosa.util.normalize(wave_data)  

    # bring audio to uniform lenth of appr. 30s
    wave_data = wave_data[:465984]

    # time-frequency transformation with Mel-scaling
    spectrogram = librosa.feature.melspectrogram(wave_data,
                                                 sr=sample_rate,
                                                 n_mels     = 80,
                                                 power      = 2.0,
                                                 norm       = 1)


    # transform to Decibel scale
    spectrogram = librosa.power_to_db(spectrogram)
    # re-shape to final segment size
    return spectrogram.astype(np.float32)
def getDataFrame(is_test):
        temp_df = None
        if is_test:
            temp_df = pd.read_csv('test.csv')
        else:
            temp_df = pd.read_csv('%s/features_30_sec.csv' % DATA_PATH)
        
        temp_df['filePath'] = DATA_PATH + '/genres_original/' + temp_df['label'] + '/' + temp_df['filename']

        ids = copy.deepcopy(temp_df['filename'])
        bits = []
        index = 0

        for id in ids:
            bits = id.split('.')
            ids[index] = 'id-'+bits[0][0:2]+bits[1]+'-original'
            index += 1
        temp_df['ID'] = ids

        return temp_df.loc[:, ['ID','filePath', 'label']]
df = getDataFrame(True)
# print(testInst.labels.shape)
# print(testInst.spectrograms.shape)
# print(testInst.spectrograms[0])
# print(testInst.labels[0])
# fig, ax = plt.subplots()
# img = librosa.display.specshow(testInst.spectrograms[0], ax=ax)
# fig.colorbar(img, ax=ax)