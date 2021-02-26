from torch import nn
from .tp import get_tp_loss

def train_model(model, config, reporter, device, loader, optimizer, epoch):
    model.train()

    reporter.reset_epoch_data()
    for batch_idx, (data, targets) in enumerate(loader):
        if config.is_tangent_prop:
            data.requires_grad = True

        predictions = get_model_prediction(model, data, targets, device, config.aug_params.segmented)
        
        loss, tp_loss = get_model_loss(predictions, targets, config, device)
        reporter.record_batch_data(predictions, targets, loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            reporter.keep_log(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item())
                )
    reporter.record_epoch_data(model, epoch)

def test_model(model, device, loader):
    model.eval()
    

def get_model_prediction(model, data, targets, device, is_segmented):
    preds_sum = None
    if is_segmented:
        preds_sum = torch.from_numpy(np.zeros((data.shape[0], 10)))
        for i in range(6):
            strip_data = data[:,i,:,:].to(device=device)
            targets = targets.to(device=device)
            preds = model(strip_data)
            preds_sum += preds
    else:
        preds_sum = model(data)
    return preds_sum


def get_model_loss(predictions, targets, config, device, x=None, transformed_data=None):
    tp_loss = 0
    if x is not None and transformed_data is not None:
        tp_loss = get_tp_loss(x, predictions, config.e0, device, transformed_data)
    
    model_loss = config.loss(predictions, targets) + config.gamma * tp_loss
    return model_loss, config.gamma * tp_loss.item()

def init_layer(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')