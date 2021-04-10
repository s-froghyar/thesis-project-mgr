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
        'train': temp_df_train.loc[:, ['filePath', 'label']].sample(frac = 1),
        'test': temp_df_test.loc[:, ['filePath', 'label']].sample(frac = 1)
    }

    return out

def generate_6_strips(wd):
    return np.array_split(wd[:465984], 6)