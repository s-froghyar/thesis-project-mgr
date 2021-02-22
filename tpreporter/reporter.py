import os, re, os.path
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import confusion_matrix
from .reporter_utils import *

possible_names = ['baseline', 'segmented']

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
        
        # my_path = os.path.join(os.getcwd(), 'data', 'reports', name)

        # if os.path.exists(my_path):
        #     for root, dirs, files in os.walk(my_path):
        #         for file in files:
        #             os.remove(os.path.join(root, file))

        self.name = name
        self.log_path = save_directory
        self.max_epochs = max_epochs

        self.train_confusion_matrix = None
        self.test_confusion_matrix = None
        self.train_summary_writer = SummaryWriter(self.log_path)

    def set_post_training_values(self, model, train_set, test_set):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
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
                                    
    
    # def show_confusion_matrix(self, train=True):
    #     if self.confusion_matrix is not None:
    #         if train:
    #             plot_confusion_matrix(self.train_confusion_matrix)
    #         else:
    #             plot_confusion_matrix(self.test_confusion_matrix)
    #     else:
    #         raise ValueError(
    #             'confusion matrix is not generated yet, please run Reporter.report_on_model() to generate it.')
    
    def record_first_batch(self, model, train_set_len, first_item):
        print('recording first batch data')

        with torch.no_grad():
            first_item = first_item.unsqueeze(0)
            self.train_set_len = train_set_len
            self.train_summary_writer.add_image('images', first_item)
            self.train_summary_writer.add_graph(model, first_item)
    def reset_epoch_data(self):
        self.total_loss = 0
        self.total_correct = 0
    def record_batch_data(self, predictions, targets, loss): # could be further extended
        self.total_loss += loss.item()
        self.total_correct += get_num_correct(predictions, targets.numpy())
    
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
    # TODO
    def keep_log(self, log_string):
        print(log_string)
    def __str__(self):
        return str(dict(
            name = self.name,
            log_path = self.log_path
        ))
        