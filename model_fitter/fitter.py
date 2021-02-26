from .dataset import *
from .utils import *
import time
from torch.utils.data import DataLoader

class ModelFitter:
    def __init__(self, args, model_config, device, kwargs, reporter):
        self.args = args
        self.model_config = model_config
        self.device = device
        self.kwargs = kwargs
        self.reporter = reporter

    def fit(self):
        model, optimizer = self.init_model()
        train_loader, test_loader = self.get_data_loaders()
        self.reporter.record_first_batch(model, len(train_loader), next(iter(train_loader))[0][0])
        
        for epoch in range(self.model_config.epochs):
            train_model(model, self.model_config, self.reporter, self.device, train_loader, optimizer, epoch)
        
    def evaluate(self):
        return self.reporter.report_on_model()



    def get_data_loaders(self):
        print("Loading in data...")
        if self.model_config.pre_augment:
            GTZAN = load_wave_data(
                self.args.data_path,
                aug_params=self.model_config.aug_params,
                is_pre_augmented=True
                self.args.local)

            train_loader = DataLoader(
                GtzanPreAugmentedDataset(
                    GTZAN.train_x,
                    GTZAN.train_y,
                    self.model_config.dataset_params,
                    train=True), 
                batch_size=self.model_config.batch_size)

            test_loader = DataLoader(
                GtzanPreAugmentedDataset(
                    GTZAN.test_x,
                    GTZAN.test_y,
                    self.model_config.dataset_params,
                    train=False),
                batch_size=self.model_config.batch_size, shuffle=True)

            return train_loader, test_loader
        else:
            GTZAN = load_wave_data(
                self.args.data_path,
                aug_params=self.model_config.aug_params,
                is_pre_augmented=False
                self.args.local)
            train_loader = DataLoader(
                GtzanDynamicDataset(
                    GTZAN.train_x,
                    GTZAN.train_y,
                    self.model_config.dataset_params,
                    self.model_config.e0,
                    self.device
                )
            )
            test_loader = DataLoader(
                GtzanDynamicDataset(
                    GTZAN.test_x,
                    GTZAN.test_y,
                    self.model_config.dataset_params,
                    self.model_config.e0,
                    self.device
                )
            )
            return train_loader, test_loader

    def init_model(self):
        out_model = self.model_config.model().to(self.device)
        out_model.apply(init_layer)
        
        out_optimizer = self.model_config.optimizer(out_model.parameters(), lr=self.model_config.lr)
        
        return out_model, out_optimizer

