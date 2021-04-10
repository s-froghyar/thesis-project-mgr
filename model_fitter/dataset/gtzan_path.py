import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import pickle

from .dataset_utils import BASE_SAMPLE_RATE, genre_mapping, get_data_frame

class GtzanPath:
    """
    GTZAN Path to Data generator
    Only wave data is the output
    Arguments:
        - df - Dataframe containing data path and label
        - test_size - [0, 1] proportion of data to be used as validation
    """
    def __init__(
        self,
        dfs,
        test_size=0.1,
    ):
        init_x = dfs['train']['filePath']
        init_y = dfs['train']['label']
        init_test_x = dfs['test']['filePath']
        init_test_y = dfs['test']['label']

        self.set_up_test_data(init_test_x, init_test_y)
        self.set_up_train_data(init_x, init_y)
        self.give_report()
    
    
    def set_up_test_data(self, init_test_x, init_test_y):
        print("Preparing test values...")
        self.test_x = []
        self.test_y = []
        for index, path in init_test_x.iteritems():            
            self.test_x.append(path)
            self.test_y.append(genre_mapping[str(init_test_y[index])])

    def set_up_train_data(self, init_x, init_y):
        print("Preparing original train data...")

        self.train_x = []
        self.train_y = []

        for index, path in init_x.iteritems():
            self.train_x.append(path)
            self.train_y.append(genre_mapping[str(init_y[index])]) 

    def __len__(self):
        return len(self.train_y)
    def __getitem__(self, index):
        return self.train_x[index]
    def give_report(self):
        # print("Data Augmentation is completed with results:")
        print(f"Training samples: {len(self.train_x)}")
        print(f"Testing samples: {len(self.test_x)}")
    def get_metadata(self):
        return dict(
            num_of_train_data=len(self.train_x),
            num_of_test_data=len(self.test_x),
            first_item=self.train_x[0]
        )

def load_path_data(data_path, test_size=0.2, is_local=True):
    '''
        Returns tuple of (GtzanPath object, number of audio files, first_wave tuple=())
    '''
    test_file_path = ""
    gtzan_waves_path = None

    test_file_path = f"{data_path}/gtzan_dynamic_test"
    test_file_exists = os.path.isfile(test_file_path)
    
    if is_local:
        dfs = get_data_frame(data_path, True)

        if test_file_exists:
            gtzan_waves_path = load_test_data(test_file_path)
        else:
            gtzan_waves_path = GtzanPath(dfs, test_size=test_size)
            save_test_data(test_file_path, gtzan_waves_path)
    else:
        print(f"Cluster execution - preparing data")
        dfs = get_data_frame(data_path, False)
        gtzan_waves_path = GtzanPath(dfs, test_size=test_size)
    
    return gtzan_waves_path


def save_test_data(test_file_path, data):
    with open(test_file_path, 'wb') as f:
        pickle.dump(data, f)
def load_test_data(test_file_path):
    with open(test_file_path, 'rb') as f:
        return pickle.load(f)