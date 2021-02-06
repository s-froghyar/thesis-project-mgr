import librosa
import os
import librosa.display
import pandas as pd
import numpy as np
import copy
import argparse

DATA_PATH = '../../../data/GTZAN'

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

def get_data_frame(is_test):
        temp_df = None
        if is_test:
            temp_df = pd.read_csv('test.csv')
        else:
            temp_df = pd.read_csv('%s/features_30_sec.csv' % DATA_PATH)
        print(DATA_PATH)
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

def update_data_path(is_cluster):
    global DATA_PATH
    if is_cluster:
        print("Running on cluster")
        DATA_PATH = "/home/2267217f/data"
    else:
        print("Running locally")
        DATA_PATH = os.path.join(os.getcwd(), 'data', 'GTZAN')

def generate_6_strips(wd):
    return np.array_split(wd[:465984], 6)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')