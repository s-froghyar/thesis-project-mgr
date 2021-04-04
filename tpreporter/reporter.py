import os, re, os.path
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from .reporter_utils import *
import numpy as np
import datetime

possible_names = ['segmented', 'tp', 'augerino']

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
    def __init__(self, name, config, save_directory):
        if name not in possible_names:
            raise ValueError('name is not recognized from possible experiment names')
        
        self.name = name
        self.log_path = save_directory
        self.config = config

        self.epoch_losses = []
        self.epoch_corrects = []
        self.epoch_accuracies = []

        self.total_loss = 0
        self.total_correct = 0
        self.augerino_losses = []
        self.tp_losses = []
        self.tta_correct = []
        self.tta_correct_val = 0

        self.create_logging_env()
        self.train_summary_writer = SummaryWriter(f"{self.log_path}/tensorboard")
        self.keep_log(
            f'''
            Reporter started with params:
            name: {self.name}
            log_path: {self.log_path}
            '''
        )

    def record_batch_data(self, predictions, targets, losses):
        loss, tp_loss, augerino_loss = losses

        self.record_specific_loss(tp_loss, augerino_loss)
        self.total_loss += loss.item()
        
        self.total_correct += get_num_correct(predictions, targets)

    def record_specific_loss(self, tp, aug):
        if self.config.model_type == 'augerino':
            self.augerino_losses.append(aug.item())
        elif self.config.model_type == 'tp':
            self.tp_losses.append(tp.item())
    def record_tta(self, preds, targets):
        num_corr = get_num_correct(preds, targets)
        self.tta_correct_val += num_corr

    def record_epoch_data(self, epoch):
        self.train_summary_writer.add_scalar("Loss", self.total_loss, epoch)
        self.train_summary_writer.add_scalar("Correct", self.total_correct, epoch)
        self.train_summary_writer.add_scalar("Accuracy", self.total_correct / self.train_set_len, epoch)

        self.epoch_losses.append(self.total_loss)
        self.epoch_corrects.append(self.total_correct)
        self.epoch_accuracies.append(self.total_correct / self.train_set_len)

        self.tta_correct.append(self.tta_correct_val)
        # self.train_summary_writer.add_histogram("conv1.bias", model.conv1.bias, epoch)
        # self.train_summary_writer.add_histogram("conv1.weight", model.conv1.weight, epoch)
        # self.train_summary_writer.add_histogram("conv2.bias", model.conv2.bias, epoch)
        # self.train_summary_writer.add_histogram("conv2.weight", model.conv2.weight, epoch)
        if (epoch-1) == self.config.epochs:
            print('report is available through tensorboard in the reports folder')
            self.train_summary_writer.close()
        self.reset_epoch_data()

    def reset_epoch_data(self):
        self.keep_log(f"Epoch finished with accuracy: {self.total_correct / self.train_set_len}")
        self.total_loss = 0
        self.total_correct = 0
        self.tta_correct_val = 0
        
    def save_metrics(self, epoch):
        torch.save({
            "epoch_losses": self.epoch_losses,
            "epoch_corrects": self.epoch_corrects,
            "epoch_accuracies": self.epoch_accuracies,
            "augerino_losses": self.augerino_losses,
            "tp_losses": self.tp_losses,
            "tta_correct": self.tta_correct
        }, f"{self.log_path}/model_metrics_e{epoch}")
    def save_predictions_for_cm(self, preds, targets, epoch):
        torch.save({
            "predictions": preds,
            "targets": targets,
        }, f"{self.log_path}/confusion_matrix_e{epoch}")
    def save_model(self, model, epoch):
        torch.save(model.state_dict(), f"{self.log_path}/e{epoch}_model.pt")
    
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
