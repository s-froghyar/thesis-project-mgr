from .dataset import load_path_data, GtzanTTADataset
from .utils import *
# from .utils import GaussianNoiseAug, PitchShiftAug, AugAveragedModel
# import time
# import torch.nn as nn
from torch.utils.data import DataLoader
import torch
# import torchaudio.transforms as aud_transforms

class Interpreter:
    '''
    Given a model, it will test it with:
        - no augmentation
        - TTA normal bounds
        - TTA augerino bounds
    '''
    def __init__(self, model_combo, aug_params, model_type, augerino_bounds=None):
        self.params = dict(
            frames=256,
            bands=128,
            window_size=1024,
            hop_size=256,
            e0=1e-3
        )
        self.aug_params = aug_params
        self.model_type = model_type
        self.device = torch.device('cpu')

        self.model = self.get_model(model_combo)

        
        # self.test_loader = self.get_loader()


    def run_evaluation(self):
        out = {}
        augs = self.get_augs()
        print('Running no augment test')
        
        # No aug
        test_loader = self.get_loader(augs[0])
        no_aug_predictions, no_aug_targets = evaluate_model(self.model, test_loader, self.model_type)
        no_aug_accuracy = get_num_correct(no_aug_predictions, no_aug_targets)
        out['no_aug'] = {
            'accuracy': no_aug_accuracy / 100,
            'predictions': no_aug_predictions,
            'targets': no_aug_targets
        }

        print('Running normal TTA')
        # TTA
        test_loader = self.get_loader(augs[1])
        tta_predictions, tta_targets = evaluate_model(self.model, test_loader, self.model_type)
        tta_accuracy = get_num_correct(tta_predictions, tta_targets)
        out['tta_normal'] = {
            'accuracy': tta_accuracy / 100,
            'predictions': tta_predictions,
            'targets': tta_targets
        }
        # # TTA custom limits
        # test_loader = self.get_loader(augs[2])
        # c_tta_predictions, c_tta_targets = evaluate_model(self.model, test_loader)
        
        return out
    def get_augs(self):
        if self.aug_params.transform_chosen == 'ni':
            return [None, (9., 12.)]
        elif self.aug_params.transform_chosen == 'ps':
            return [None, (-2., 2.)]
        else:
            return [None]

    
    
    def get_model(self, combo):
        net = combo[0]()
        if self.model_type != 'augerino': 
            net.load_state_dict(combo[1])
            return net

        aug = None
        if self.aug_params.transform_chosen == 'ps': aug = PitchShiftAug()
        elif self.aug_params.transform_chosen == 'ni': aug = GaussianNoiseAug()
        else: 
            raise RuntimeError('no augmentation for augerino... a bit counterintuitive dont u think?')

        model = AugAveragedModel(net, aug, get_model_prediction, self.device)
        model.load_state_dict(combo[1])

        return model




    def get_loader(self, augs):
        GTZAN = load_path_data(
            'data',
            test_size=0.1,
            is_local=False
        )
        return DataLoader(
                GtzanTTADataset(
                    paths           = GTZAN.test_x,
                    labels          = GTZAN.test_y,
                    mel_spec_params = self.params,
                    aug_params      = self.aug_params,
                    device          = self.device,
                    train           = False,
                    tta_settings    = augs
                ),
                batch_size=16,
                shuffle=False
            )