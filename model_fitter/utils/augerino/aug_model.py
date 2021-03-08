import torch
import torch.nn as nn


class AugAveragedModel(nn.Module):
    def __init__(self, model, aug,ncopies=4):
        super().__init__()
        self.aug = aug.double()
        self.model = model.double()
        self.ncopies = ncopies
    def forward(self, x):
        if self.training: 
            return self.model(self.aug(x))
        else: 
            bs = x.shape[0]
            aug_x = torch.cat([self.aug(x) for _ in range(self.ncopies)],dim=0)
            return sum(torch.split(F.log_softmax(self.model(aug_x),dim=-1),bs))/self.ncopies