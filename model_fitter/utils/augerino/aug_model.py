import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as aud_transforms

class AugAveragedModel(nn.Module):
    def __init__(self, model, aug, pred_getter, device, ncopies=4):
        super().__init__()
        self.aug = aug.double()
        self.model = model.double()
        self.ncopies = ncopies
        self.device = device
        self.pred_getter = pred_getter
    def forward(self, x):
        if self.training:
            transform = nn.Sequential( 
                    aud_transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_mels=128,
                    n_fft=1024,
                    hop_length=256
                ).to(self.device),
                aud_transforms.AmplitudeToDB().to(self.device)

            ).double().to(self.device)
            num_of_patches = x.size(1)
            all_specs = []
            for i in range(num_of_patches):
                batch_of_patches = x[:,i,:].to(self.device)
                augmentations = self.aug(batch_of_patches).to(self.device)
                mel_specs = transform(augmentations).to(self.device)
                all_specs.append(mel_specs)
            all_specs = torch.stack(all_specs).permute((1,0,2,3))
            preds_sum = torch.from_numpy(np.zeros((all_specs.size(0), 10))).to(dtype=torch.float32, device=self.device)

            for i in range(num_of_patches):
                strip_data = all_specs[:,i,:,:]
                strip_data.requires_grad_(True)
                preds = self.model(strip_data)
                preds_sum += preds.to(device=self.device)
            return preds_sum
        else:
            return self.model(x.double())


def splitsongs(wd, overlap = 0.0):
    stacked = []
    for i in range(wd.size(0)):
        temp_X = []

        # Get the input song array size
        xshape = wd.shape[0]
        chunk = 20000
        offset = int(chunk*(1.-overlap))

        # Split the song and create new ones on windows
        spsong = [wd[i, j:j+chunk] for j in range(0, xshape - chunk + offset, offset)]
        for s in spsong:
            if s.shape[0] != chunk:
                continue

            temp_X.append(s)
        stacked.append(temp_X)

    return stacked