import os, re, os.path
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import confusion_matrix
from .reporter_utils import *
import numpy as np
import datetime

possible_names = ['baseline', 'segmented', 'tp', 'augerino']

class Reporter():
    """
        Common data collector for all models

        Use steps:
            1) Before training: 
                a) init with one of the predefined names and number of epochs for training
                    # reporter = Reporter('baseline', num_epochs)
                b) optionally record first batch data for tensorboard model graph and first spectrogram
                    # reporter.record_first_batch(model, train_loader)
            2) During training:
                a) reset epoch data at start of the epoch
                    # reporter.reset_epoch_data()
                b) record batch data in each batch
                    # reporter.record_batch_data(preds, targets, loss)
                c) record epoch data at the end of each epoch
                    # reporter.record_epoch_data(model, epoch)
            3) After training:
                a) set post training values --> model and train_loader
                    # reporter.set_post_training_values(model, train_loader)
                b) report on model to generate confusion matrices and accuracies
                    # train_num_correct, test_num_correct = reporter.report_on_model()
    """
    def __init__(self, name, max_epochs, save_directory):
        if name not in possible_names:
            raise ValueError('name is not recognized from possible experiment names')
        
        self.name = name
        self.log_path = save_directory
        self.max_epochs = max_epochs

        self.train_confusion_matrix = None
        self.test_confusion_matrix = None
        self.create_logging_env()
        self.train_summary_writer = SummaryWriter(f"{self.log_path}/tensorboard")
        self.keep_log(
            f'''
            Reporter started with params:
            name: {self.name}
            log_path: {self.log_path}
            '''
        )

    def set_post_training_values(self, model, train_set, test_set):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
    
    
    def record_first_batch(self, model, train_set_len, first_item):
        print('recording first batch data')
        
        with torch.no_grad():
            if not isinstance(first_item, np.ndarray):
                first_item = first_item.unsqueeze(0)
            # self.train_summary_writer.add_image('images', first_item)
            self.train_set_len = train_set_len
            # self.train_summary_writer.add_graph(model, first_item)
    def reset_epoch_data(self):
        self.total_loss = 0
        self.total_correct = 0
    def record_batch_data(self, predictions, targets, loss): # could be further extended
        self.total_loss += loss
        batch_correct = get_num_correct(predictions, targets)
        self.total_correct += batch_correct
        return batch_correct
    
    def record_epoch_data(self, model, epoch):
        self.train_summary_writer.add_scalar("Loss", self.total_loss, epoch)
        self.train_summary_writer.add_scalar("Correct", self.total_correct, epoch)
        self.train_summary_writer.add_scalar("Accuracy", self.total_correct / self.train_set_len, epoch)

        self.train_summary_writer.add_histogram("conv1.bias", model.conv1.bias, epoch)
        self.train_summary_writer.add_histogram("conv1.weight", model.conv1.weight, epoch)
        self.train_summary_writer.add_histogram("conv2.bias", model.conv2.bias, epoch)
        self.train_summary_writer.add_histogram("conv2.weight", model.conv2.weight, epoch)
        if (epoch-1) == self.max_epochs:
            print('report is available through tensorboard in the reports folder')
            self.train_summary_writer.close()
    def save_model(self, model):
        torch.save(model.state_dict(), f"{self.log_path}/final_model.pt")
    def keep_log(self, log_string):
        log_text = f"\n{str(datetime.datetime.now())}: {log_string}"
        with open(f"{self.log_path}/logs", 'a') as f:
            f.write(log_text)
            print(log_text)

    def create_logging_env(self):
        full_data_path = os.path.join(os.getcwd(), self.log_path)
        try:
            dir_content = os.listdir(full_data_path)
            self.log_path = f"{self.log_path}/run_{len(dir_content) + 1}"
        except FileNotFoundError:
            os.mkdir(full_data_path)
            self.log_path = f"{self.log_path}/run_1"
    def __str__(self):
        return str(dict(
            name = self.name,
            log_path = self.log_path
        ))
