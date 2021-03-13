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
    temp_df = None
    if is_local:
        temp_df = pd.read_csv(f"{data_path}/test.csv")
    else:
        temp_df = pd.read_csv(f"{data_path}/features_30_sec.csv")

    temp_df['filePath'] = data_path + '/' + temp_df['label'] + '/' + temp_df['filename']

    ids = copy.deepcopy(temp_df['filename'])

    for index, id in enumerate(ids):
        bits = id.split('.')
        ids[index] = f"id-{bits[0][0:2]}{bits[1]}-original"
    temp_df['ID'] = ids

    return temp_df.loc[:, ['ID','filePath', 'label']], len(temp_df)

def generate_6_strips(wd):
    return np.array_split(wd[:465984], 6)