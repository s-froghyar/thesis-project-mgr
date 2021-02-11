from .models import BaselineCNN, SegmentedCNN, TpCNN, AugerinoCNN
import torch
import torch.optim as optim

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
    propagate_tangent = False
    test_transform = None
    test_e0 = 0
    audio_transform = None

class BaselineCNNconfig(CNNconfig):
    model = BaselineCNN
    aug_params = dict(
        segmented=False,
        noise_injection=(0.0, 0.1, 0.02),
        pitch_shift=(-5, 0, 1)
    )
class SegmentedCNNconfig(CNNconfig):
    model = SegmentedCNN
    aug_params = dict(
        segmented=True,
        noise_injection=(0.0, 0.1, 0.02),
        pitch_shift=(-5, 0, 1)
    )
class TpCNNconfig(CNNconfig):
    model = TpCNN
    aug_params = dict(
        segmented=False,
        noise_injection=(0.0, 0.1, 0.02),
        pitch_shift=(-5, 0, 1)
    )
class AugerinoCNNconfig(CNNconfig):
    model = AugerinoCNN
    aug_params = dict(
        segmented=False,
        noise_injection=(0.0, 0.1, 0.02),
        pitch_shift=(-5, 0, 1)
    )
