import os
import pandas as pd
import numpy as np
import pickle
import copy
from .gtzan_wave import GtzanWave


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

def load_wave_data(data_path, aug_params=None, is_pre_augmented=True, is_local=True):
    test_file_path = ""
    if is_pre_augmented:
        test_file_path = f"{data_path}/gtzan_augmented_test"
    else:
        test_file_path = f"{data_path}/gtzan_dynamic_test"
    
    test_file_exists = os.path.isfile(test_file_path)
    
    if is_local:
        if test_file_exists:
            with open(test_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            df = get_data_frame(data_path, True)
            temp = GtzanWave(df, pre_augment=is_pre_augmented, aug_params=aug_params)
            with open(test_file_path, 'wb') as f:
                pickle.dump(temp, f)
            return temp
    else:
        df = get_data_frame(data_path, False)
        return GtzanWave(df, pre_augment=is_pre_augmented, aug_params=aug_params)

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

    return temp_df.loc[:, ['ID','filePath', 'label']]

def get_correct_input_format(wave_data, is_segmented):
    if is_segmented:
        return generate_6_strips(wave_data)
    else:
        return wave_data[:465984]

def generate_6_strips(wd):
    return np.array_split(wd[:465984], 6)