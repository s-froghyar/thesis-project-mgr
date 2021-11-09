from torch import nn
import numpy as np
import torch
import torchaudio.transforms as aud_transforms
import matplotlib.pyplot as plt

from .utils import *

def test_model(model, config, reporter, device, loader, epoch):
    ''' TTA '''
    model.eval()
    chosen_aug = config.aug_params.transform_chosen

    all_targets = None
    all_predictions = None
    
    with torch.no_grad():
        for batch_idx, (base_data, targets) in enumerate(loader):
            base_data, targets = base_data.to(device), targets.to(device)
            n_augs = 4
            if base_data.size(1) == 1:
                n_augs = 1
            preds = [get_model_prediction(model, base_data[:,i,:,:,:], device, config.model_type, is_eval=True) for i in range(n_augs)]
            final_predictions = get_final_preds(preds, device)

            reporter.record_tta(final_predictions.to(device), targets.to(device))

            if all_predictions is None:
                all_predictions = final_predictions
            else:
                all_predictions = torch.vstack((all_predictions, final_predictions))
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets))

    return all_predictions, all_targets


def report_on_model(self):
    print('\n\Getting report on model...')
    train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64)
    
    self.train_predictions = get_all_preds(self.model, train_loader)
    self.test_predictions = get_all_preds(self.model, test_loader)

    train_num_correct = get_num_correct(self.train_predictions, self.train_set.targets)
    test_num_correct = get_num_correct(self.test_predictions, self.test_set.targets)
    
    self.train_confusion_matrix = confusion_matrix(
                                self.train_set.targets,
                                self.train_predictions.argmax(dim=1))
    self.test_confusion_matrix = confusion_matrix(
                                self.test_set.targets,
                                self.test_predictions.argmax(dim=1))
    
    return (train_num_correct, test_num_correct)


def gather_outputs(model, dataset, transform_chosen, model_type):
    model.eval()
    device = torch.device('cpu')
    with torch.no_grad():
        for i in range(len(dataset)):
            data, target, aug_options = dataset[i]
            print(f"Aug option: {aug_options[0]}")
            pred, imgs = model(data[0,5,:,:].to(device))

            m = nn.Softmax()
            pred = m(pred)

            # save model outputs
            save_path = f"visualisations/network_explr/{target}/{model_type}/{transform_chosen.upper()}"
            # input
            save_image(f"{save_path}/inpatch.jpeg", imgs[0])
            # conv layers
            for i in range(12):
                save_image(f"{save_path}/conv1/conv1_fm_{i}.jpeg", imgs[1][i,:,:])
                save_image(f"{save_path}/conv2/conv2_fm_{i}.jpeg", imgs[2][i,:,:])
            print(data.shape)
            line1 = [sum(imgs[3][ind:(ind+50)]) for ind in range(10)]

            with open(f"{save_path}/arrs.txt", "w") as f:

                f.writelines((str(line1), str(list(pred)), str(aug_options[0])))



def save_image(name, img):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.matshow(img)
    plt.savefig(name, dpi = height) 
    plt.close()
def evaluate_model(model, loader, model_type):
    model.eval()

    device = torch.device('cpu')
    all_targets = None
    all_predictions = None
    with torch.no_grad():
        for batch_idx, (base_data, targets) in enumerate(loader):
            base_data, targets = base_data.to(device), targets.to(device)
            n_augs = 4
            if base_data.size(1) == 1:
                n_augs = 1
            preds = [get_model_prediction(model, base_data[:,i,:,:,:], device, model_type, is_eval=True) for i in range(n_augs)]
            final_predictions = get_final_preds(preds, device)

            if all_predictions is None:
                all_predictions = final_predictions
            else:
                all_predictions = torch.vstack((all_predictions, final_predictions))
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets))
    return all_predictions, all_targets