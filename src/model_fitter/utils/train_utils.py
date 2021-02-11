from torch import nn

def train_model(model, config, reporter, device, loader, optimizer, epoch):
    # preds_sum = torch.from_numpy(np.zeros((data.shape[0], 10)))
    # for i in range(6):
    #     # Get data to cuda if possible
    #     strip_data = data[:,i,:,:].to(device=device)
    #     targets = targets.to(device=device)
    #     preds = model(strip_data)
    #     preds_sum += preds
    model.train()
    reporter.reset_epoch_data()
    for batch_idx, (data, targets) in enumerate(loader):
        predictions = get_model_prediction(config.aug_params["segmented"])
        
        loss = get_model_loss(predictions, targets, config, device)
        
        reporter.record_batch_data(predictions, targets, loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            reporter.keep_log(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), training_loss.item())
            )
    reporter.record_epoch_data(model, epoch)

def test_model(model, device, loader):
    model.eval()
    

def get_model_prediction(model, data, targets, device, is_segmented):
    if is_segmented:
        preds_sum = torch.from_numpy(np.zeros((data.shape[0], 10)))
        for i in range(6):
            # Get data to cuda if possible
            strip_data = data[:,i,:,:].to(device=device)
            targets = targets.to(device=device)
            preds = model(strip_data)
            preds_sum += preds


# TODO
def get_model_loss(predictions, targets, config, device):
    return config.loss(predictions, targets)

def init_layer(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')