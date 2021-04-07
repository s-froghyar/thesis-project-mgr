import torch
import torch.nn as nn
import numpy as np
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
            # aug gens full spec from wave data, we need to split and make sum prediction
            # full_spec = self.aug(x)
            
            # patches = torch.split(full_spec, 76, dim=2)
            # split_specs = torch.stack(patches).permute(1,0,2,3)
            print(x.shape)
            patches = splitsongs(x)
            mel_specs = [self.aug(patch) for patch in patches] # 456 width
            print(len(mel_specs))
            split_specs = torch.stack(mel_specs)
            preds_sum = torch.from_numpy(np.zeros((split_specs.shape[0], 10))).to(dtype=torch.float32, device=self.device)

            for i in range(27):
                strip_data = split_specs[:,i,:,:]
                strip_data.requires_grad_(True)
                preds = self.model(strip_data)
                preds_sum += preds.to(device=self.device)
            return preds_sum
        else:
            return self.model(x.double())


def splitsongs(wd, overlap = 0.5):
    temp_X = []

    # Get the input song array size
    xshape = wd.shape[0]
    chunk = 33000
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [wd[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)