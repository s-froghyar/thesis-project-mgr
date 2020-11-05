import seaborn as sns
import os
import glob
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
# REAL DATA
# DATA_PATH = 'GTZAN'
# df = pd.read_csv('%s/features_30_sec.csv' % DATA_PATH)
# df['filePath'] = DATA_PATH + '/genres_original/' + df['label'] + '/' + df['filename']

# TEST DATA
df = pd.read_csv('test.csv')
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
        - time-stretch: (0.8, 1.2, step=0.02)
        - patches_per_spectrogram: 1 (should we? probably not)
    """
    def __init__(self, noise_injection=(0.0, 0.1, 0.02)):
        self.initDataFrame(noise_injection)

    def initDataFrame(self, noise_injection):
        # calculate length of tensors to hold data
        DATA_LENGTH = len(df)

        NOISE_INJECTION_STEPS = ((noise_injection[1] - noise_injection[0]) / noise_injection[2]) - 1
        NUM_OF_AUGMENTED_DATA = (DATA_LENGTH) * NOISE_INJECTION_STEPS

        FULL_DATA_LENGTH = int(DATA_LENGTH + NUM_OF_AUGMENTED_DATA)
        print('data length is %i' % (DATA_LENGTH))
        print('Noise injections: %i' % (NOISE_INJECTION_STEPS))
        print('Creating %i spectrograms and labels' % (FULL_DATA_LENGTH))
        print('NI:' + str(noise_injection))

        self.spectrograms = np.full((FULL_DATA_LENGTH, 80, 911), -1)
        self.labels = np.full((FULL_DATA_LENGTH), -1)

        for index, row in tqdm(df.iterrows()):
            fileName = row['filePath']
            wave_data, sample_rate = librosa.core.load(fileName, 
                                               sr    = None, 
                                               mono  = True, 
                                               dtype = np.float32)
            # We have the wave data now lets augment it -> Noise injection first
            print('Occupying: %i' % (int(index + index * NOISE_INJECTION_STEPS)))
            self.create_noise_injected_data(wave_data,
                                        sample_rate,
                                        noise_injection,
                                        int(index + index * NOISE_INJECTION_STEPS),
                                        genre_mapping[str(row['label'])])
            self.spectrograms[int(index + index * NOISE_INJECTION_STEPS)] = create_spectrogram(wave_data, sample_rate, 16000)
            self.labels[int(index * NOISE_INJECTION_STEPS)] = genre_mapping[str(row['label'])]
    
    
    def create_noise_injected_data(self, wd, sr, ni, pivot, label):
        step = 1
        print(str(ni))
        for noise_factor in np.arange(ni[0] + ni[2], ni[1] + ni[2], ni[2]):
            print('noise factor: %f' % (noise_factor))
            print('index occupied: %i' % (int(pivot + step)))
            noise = np.random.randn(len(wd))
            augmented_data = wd + noise_factor * noise
            # # Cast back to same data type
            self.spectrograms[int(pivot + step)] = create_spectrogram(augmented_data.astype(type(wd[0])), sr, 16000)
            self.labels[int(pivot + step)] = label
            step += 1
            

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

testInst = GtzanData()
print(testInst.labels.shape)
print(testInst.spectrograms.shape)
print(testInst.spectrograms[0])
print(testInst.labels[0])