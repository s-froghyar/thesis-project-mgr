from .models import SegmentedCNN
from .aug_params import AugmentationParameters
import torchaudio.transforms as aud_transforms
import torch
import torch.optim as optim
class CNNconfig:
    batch_size = 16
    epochs = 10
    lr = 0.01
    seed = 1
    log_interval = 1
    save_model = True
    loss = torch.nn.CrossEntropyLoss()
    segmented = False
    test_size = 0.2
    BASE_SAMPLE_RATE = 16000
    model = SegmentedCNN
    local = False

    dataset_params = dict(
        frames=256,
        bands=128,
        window_size=1024,
        hop_size=256,
        e0=1e-3
    )
    mel_spec_transform = aud_transforms.MelSpectrogram(
                sample_rate=BASE_SAMPLE_RATE,
                n_mels=dataset_params["bands"],
                n_fft=dataset_params["window_size"],
                hop_length=dataset_params["hop_size"]
            )

    aug_params = AugmentationParameters(
        ni = (0.0, 0.2, 0.02),
        ps = [-12, 0, 12]
    )
    tta_settings = {
        'ni': (0.0, 0.2),
        'ps': (-12., 12.)
    }
    optimizer = optim.Adam
    weight_decay = 5e-4
    is_tangent_prop = False
    test_transform = None
    augerino = False
    mel_spec_transform = aud_transforms.MelSpectrogram(
                sample_rate=BASE_SAMPLE_RATE,
                n_mels=dataset_params["bands"],
                n_fft=dataset_params["window_size"],
                hop_length=dataset_params["hop_size"]
            )

class SegmentedCNNconfig(CNNconfig):
    model_type = 'segmented'
    gamma = 0
    segmented = True

class TpCNNconfig(CNNconfig):
    model_type = 'tp'
    batch_size = 4
    gamma = 0.0005
    e0=1e-3
    segmented = False
    is_tangent_prop = True
    

class AugerinoCNNconfig(CNNconfig):
    model_type = 'augerino'
    segmented = True
    is_tangent_prop = False
    augerino = True

