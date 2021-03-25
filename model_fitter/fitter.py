from .dataset import *
from .utils import *
from .utils import GaussianNoiseAug, PitchShiftAug, AugAveragedModel
import time
from torch.utils.data import DataLoader
import torchaudio.transforms as aud_transforms

class ModelFitter:
    def __init__(self, args, model_config, device, kwargs, reporter):
        self.args = args
        self.model_config = model_config
        self.device = device
        self.kwargs = kwargs
        self.reporter = reporter

        dataset_params = self.model_config.dataset_params
        self.spectrogram_transform = aud_transforms.MelSpectrogram(
                sample_rate=BASE_SAMPLE_RATE,
                n_mels=dataset_params["bands"],
                n_fft=dataset_params["window_size"],
                hop_length=dataset_params["hop_size"]
            )


    def fit(self): 
        model, optimizer = self.init_model()
        train_loader, test_loader, data_count, metadata = self.get_data_loaders()

        self.reporter.train_set_len = len(train_loader.dataset)
        
        for epoch in range(self.model_config.epochs):
            train_model(model, self.model_config, self.reporter, self.device, train_loader, optimizer, epoch)
            test_model(model, self.model_config, self.reporter, self.device, test_loader, self.spectrogram_transform)
            self.reporter.record_epoch_data(epoch)
        self.reporter.save_model(model)
        self.reporter.save_metrics()

        self.evaluate(model, test_loader)


    def evaluate(self, model, loader):
        self.reporter.keep_log('Evaluation has started')
        # confusion matrix needs all targets and predictions
        model = model.float()
        model.eval()
        chosen_aug = self.model_config.aug_params.transform_chosen
        all_preds = None
        all_targets = None
        tta = None
        if chosen_aug == 'ni':
            tta = nn.Sequential(GaussianNoiseAug(), self.spectrogram_transform.to(torch.float32))
        else:
            tta = nn.Sequential(PitchShiftAug(), self.spectrogram_transform.to(torch.float32))
        tta = tta.double()
        with torch.no_grad():
            for batch_idx, (base_data, transformed_data, augmentations, waveforms, targets) in enumerate(loader):
                new_targets = []
                for t in targets:
                    for _ in range(6):
                        new_targets.append(t)
                new_targets = torch.tensor(new_targets)
                for i in range(6):
                    data = None
                    if self.model_config.model_type == 'augerino':
                        data = waveforms[:,i,:]
                    else:
                        data = tta(waveforms[:,i,:].double())
                    predictions = model(data)
                    if all_preds is None:
                        all_preds = predictions
                    else:
                        all_preds = torch.vstack((all_preds, predictions))

        
                    self.reporter.record_tta(predictions, targets)
                if all_targets is None:
                    all_targets = new_targets
                else:
                    all_targets = torch.vstack((all_targets, new_targets))
        self.reporter.save_predictions_for_cm(all_preds, all_targets)




    def get_data_loaders(self):
        print("Loading in data...")
        
        GTZAN, data_count = load_path_data(
            self.args.data_path,
            test_size=self.model_config.test_size,
            is_local=self.args.local
        )
        train_loader = DataLoader(
            GtzanDynamicDataset(
                paths           = GTZAN.train_x,
                labels          = GTZAN.train_y,
                mel_spec_params = self.model_config.dataset_params,
                aug_params      = self.model_config.aug_params,
                device          = self.device,
                train           = True,
                model_type        = self.model_config.model_type
            ), 
            batch_size=self.model_config.batch_size
        )

        test_loader = DataLoader(
            GtzanDynamicDataset(
                paths           = GTZAN.test_x,
                labels          = GTZAN.test_y,
                mel_spec_params = self.model_config.dataset_params,
                aug_params      = self.model_config.aug_params,
                device          = self.device,
                train           = False,
                model_type        = self.model_config.model_type
            ),
            batch_size=self.model_config.batch_size,
            shuffle=True
        )
        self.reporter.keep_log(str(GTZAN.get_metadata()))
        return train_loader, test_loader, data_count, GTZAN.get_metadata()

    def init_model(self):
        if self.model_config.augerino:
            return self.init_augerino_model()
        out_model = self.model_config.model().to(device=self.device, dtype=torch.float32)
        out_model.apply(init_layer)
        
        out_optimizer = self.model_config.optimizer(out_model.parameters(), weight_decay=self.model_config.weight_decay, lr=self.model_config.lr)
        return out_model, out_optimizer

    def init_augerino_model(self):
        net = self.model_config.model().to(self.device)
        net.apply(init_layer)

        chosen_augs = []
        if self.model_config.aug_params.transform_chosen == 'ps': chosen_augs = [PitchShiftAug()]
        elif self.model_config.aug_params.transform_chosen == 'ni': chosen_augs = [GaussianNoiseAug()]
        else: chosen_augs = [GaussianNoiseAug(), PitchShiftAug()]

        aug = nn.Sequential(*tuple(chosen_augs), self.spectrogram_transform.float())
        self.model_config.model = AugAveragedModel(net, aug)

        out_optimizer = self.model_config.optimizer(self.model_config.model.parameters(), lr=self.model_config.lr)
        return self.model_config.model, out_optimizer

