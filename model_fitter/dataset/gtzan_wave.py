import numpy as np
import librosa
import librosa.display
from .dataset_utils import BASE_SAMPLE_RATE, genre_mapping, get_correct_input_format

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
        df,
        pre_augment=False,
        aug_params=None,
        test_size=0.1,
    ):
        if pre_augment:
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
        self.set_up_buckets(init_x, init_y)
        self.augment_data(init_x, init_y)
        
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)

    def augment_data(self, init_x, init_y):
        noise_injection = self.aug_params.noise_injection
        pitch_shift = self.aug_params.pitch_shift

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

