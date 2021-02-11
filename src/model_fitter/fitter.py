from .dataset import *
from .utils import *
import time

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

        for epoch in range(self.model_config.epochs):
            train_model(model, self.model_config, self.reporter, self.device, train_loader, optimizer, epoch)
        
        self.reporter.set_post_training_values(model, train_dataset, test_dataset)
    def evaluate(self):
        return self.reporter.report_on_model()



    def get_data_loaders(self):
        # Load Data
        GTZAN = load_data(self.args.data_path, self.model_config.aug_params, self.args.local)

        train_dataset = GtzanDataset(GTZAN.train_x, GTZAN.train_y, self.model_config.dataset_params, train=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.model_config.batch_size)

        test_dataset = GtzanDataset(GTZAN.test_x, GTZAN.test_y, self.model_config.dataset_params, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.model_config.batch_size, shuffle=True)

        return train_loader, test_loader

    def init_model(self):
        out_model = self.model_config.model().to(self.device)
        out_model.apply(init_layer)
        
        out_optimizer = self.model_config.optimizer(out_model.parameters(), lr=self.model_config.lr)
        
        return out_model, out_optimizer

