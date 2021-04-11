import os
import pandas as pd
import numpy as np
import pickle
import copy


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
    'rock': 9
}

def get_data_frame(data_path, is_local):
    temp_df_train = None
    temp_df_test = None

    if is_local:
        temp_df_train = pd.read_csv(f"{data_path}/local_train.csv")
        temp_df_test = pd.read_csv(f"{data_path}/local_test.csv")
        
        temp_df_train['filePath'] = data_path + '/WAV/' + temp_df_train['label'] + '/norm/' + temp_df_train['filename'].str[:-2] + 'wav'
        temp_df_test['filePath'] = data_path + '/WAV/' + temp_df_test['label'] + '/norm/' + temp_df_test['filename'].str[:-2] + 'wav'
    else:
        temp_df_train = pd.read_csv(f"{data_path}/features_30_sec_train.csv")
        temp_df_test = pd.read_csv(f"{data_path}/features_30_sec_test.csv")
        
        temp_df_train['filePath'] = data_path + '/' + temp_df_train['label'] + '/norm/' + temp_df_train['filename'].str[:-2] + 'wav'
        temp_df_test['filePath'] = data_path + '/' + temp_df_test['label'] + '/norm/' + temp_df_test['filename'].str[:-2] + 'wav'
    out = {
        'train': temp_df_train.loc[:, ['filePath', 'label']],
        'test': temp_df_test.loc[:, ['filePath', 'label']]
    }

    return out

def splitsongs(wd, overlap = 0.0):
    temp_X = []

    # Get the input song array size
    xshape = wd.shape[0]
    chunk = 20000 # min wave arr len is 478.912 --> 12 chunks (128x188) with overlap (48000)
    offset = int(chunk*(1.-overlap))

    # Split the song and create new ones on windows
    spsong = [wd[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return temp_X
