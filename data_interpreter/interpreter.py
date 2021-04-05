import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse



parser = argparse.ArgumentParser(description='Plot confusion matrix')

parser.add_argument('--epoch', '-e', required=True, choices=[str(x*5) for x in range(11)], help='The epoch checkpoint to display')

def main():
    date = '2021-04-04'
    run_id = '1'
    
    args = parser.parse_args()
    run_path = f"{date}/run_{run_id}"
    cm = load_confusion_matrix(run_path, args.epoch)
    cm['predictions'] = cm['predictions'].argmax(dim=1)
    cm['targets'] = cm['targets'].view(-1)
    cm = confusion_matrix(cm['targets'], cm['predictions'])
    plot_confusion_matrix(cm, normalize=False)


def load_confusion_matrix(run_path, poch):
    return torch.load(f"{run_path}/confusion_matrix_e{poch}", map_location=torch.device('cpu'))

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    classes = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('lmao.png')

if __name__ == "__main__":    
    main()
   # def show_confusion_matrix(self, train=True):
    #     if self.confusion_matrix is not None:
    #         if train:
    #             plot_confusion_matrix(self.train_confusion_matrix)
    #         else:
    #             plot_confusion_matrix(self.test_confusion_matrix)
    #     else:
    #         raise ValueError(
    #             'confusion matrix is not generated yet, please run Reporter.report_on_model() to generate it.')