import torch
import itertools
import numpy as np

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

