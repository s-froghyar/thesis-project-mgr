import torch
import torch.nn as nn
class AugAveragedModel(nn.Module):
    def __init__(self, model, aug, ncopies=4):
        super().__init__()
        self.aug = aug
        self.model = model
        self.ncopies = ncopies
    def forward(self, x):
        if self.training:
            return self.model(self.aug(x))
        else:
            return self.model(x)
            # bs = x.shape[0]
            # aug_x = torch.cat([self.aug(x.float()) for _ in range(4)], dim=0).float()
            # sm = nn.LogSoftmax(dim=-1)
            # return sum(torch.split(sm(self.model(aug_x)),bs))/4