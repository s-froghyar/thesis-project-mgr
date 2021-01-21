from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter

possible_names = ['baseline', 'segmented']

class Reporter():
    def __init__(name):
        if name not in possible_names:
            raise ValueError('name is not recognized from possible experiment names')

        self.train_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/train')
        self.test_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/test')

    def set_post_training_values(self, model, train_set, test_set):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
    def report_on_model():
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64)
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64)
        
        self.train_predictions = get_all_preds(self.model, train_loader)
        self.test_predictions = get_all_preds(self.model, test_loader)

        self.confusion_matrix = self.get_confusion_matrix()
    def get_confusion_matrix(self):
        print('yeet')


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

def _get_num_correct(self, preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()