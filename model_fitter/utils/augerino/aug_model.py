import torch
import torch.nn as nn
import numpy as np
class AugAveragedModel(nn.Module):
    def __init__(self, model, aug, pred_getter, ncopies=4):
        super().__init__()
        self.aug = aug.double()
        self.model = model.double()
        self.ncopies = ncopies
        self.pred_getter = pred_getter
    def forward(self, x):
        if self.training:
            # aug gens full spec from wave data, we need to split and make sum prediction
            full_spec = self.aug(x)
            
            patches = torch.split(full_spec, 76, dim=2)
            split_specs = torch.stack(patches).permute(1,0,2,3)
            preds_sum = torch.from_numpy(np.zeros((split_specs.shape[0], 10))).to(dtype=torch.float32)

            for i in range(6):
                strip_data = split_specs[:,i,:,:]
                strip_data.requires_grad_(True)
                preds = self.model(strip_data)
                preds_sum += preds
            return preds_sum
        else:
            return self.model(x.double())