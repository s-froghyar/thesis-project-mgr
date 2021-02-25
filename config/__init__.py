from .models import BaselineCNN, SegmentedCNN, TpCNN, AugerinoCNN
import torch
import torch.optim as optim

class AugmentationParameters:
    def __init__(self, ni, ps, is_segmented):
        self.noise_injection = ni
        self.pitch_shift = ps
        self.segmented = is_segmented
    def __str__(self):
        return str(dict(
            noise_injection = self.noise_injection,
            pitch_shift = self.pitch_shift,
            segmented = self.segmented
        ))
class CNNconfig:
    batch_size = 64
    test_batch_size = 64
    epochs = 50
    lr = 0.0005
    seed = 1
    log_interval = 30
    save_model = True
    loss = torch.nn.CrossEntropyLoss()

    dataset_params = dict(
        frames=128,
        bands=128,
        window_size=1024,
        hop_size=1024
    )

    optimizer = optim.Adam
    is_tangent_prop = False
    test_transform = None
    test_e0 = 0

class BaselineCNNconfig(CNNconfig):
    model = BaselineCNN
    aug_params = AugmentationParameters( (0.0, 0.1, 0.02), (-5, 0, 1), False )

class SegmentedCNNconfig(CNNconfig):
    model = SegmentedCNN
    aug_params = AugmentationParameters( (0.0, 0.1, 0.02), (-5, 0, 1), True )

class TpCNNconfig(CNNconfig):
    model = TpCNN
    aug_params = AugmentationParameters( (0.0, 0.1, 0.02), (-5, 0, 1), True )
    is_tangent_prop = True
    e0 = 1e-3

class AugerinoCNNconfig(CNNconfig):
    model = AugerinoCNN
    aug_params = AugmentationParameters( (0.0, 0.1, 0.02), (-5, 0, 1), True )

