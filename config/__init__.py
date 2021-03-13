from .models import BaselineCNN, SegmentedCNN, TpCNN, AugerinoCNN
from .aug_params import AugmentationParameters

import torch
import torch.optim as optim
class CNNconfig:
    batch_size = 5
    epochs = 50
    lr = 0.01
    seed = 1
    log_interval = 30
    save_model = True
    loss = torch.nn.CrossEntropyLoss()
    segmented = False
    test_size = 0.2


    aug_params = AugmentationParameters(
        ni = (0.0, 0.2, 0.02),
        ps = [-12, 12]
    )
    dataset_params = dict(
        frames=256,
        bands=256,
        window_size=1024,
        hop_size=1024,
        e0=1e-3
    )

    optimizer = optim.Adam
    is_tangent_prop = False
    test_transform = None
    augerino = False

class BaselineCNNconfig(CNNconfig):
    model = BaselineCNN

class SegmentedCNNconfig(CNNconfig):
    model_type = 'segmented'
    model = SegmentedCNN
    gamma = 0
    segmented = True

class TpCNNconfig(CNNconfig):
    model_type = 'tp'
    model = TpCNN
    pre_augment = False
    segmented = False
    is_tangent_prop = True
    

class AugerinoCNNconfig(CNNconfig):
    model_type = 'augerino'
    model = AugerinoCNN
    segmented = True
    is_tangent_prop = False
    pre_augment = False
    augerino = True

