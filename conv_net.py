# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from sklearn.model_selection import train_test_split
import numpy as np

from gtzan import GtzanData
from gtzan import GtzanDataset

class Config:  
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
  def __str__(self):
    output = ""
    for name, var in vars(self).items(): #Iterate over the values
      output += name + ": " + str(var) + "\n"
    return output

# Simple CNN

class CNN(nn.Module):
    def __init__(self, name, use_tensorboard=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        if use_tensorboard:
          self.train_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/train')
          self.test_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/test')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 1
num_epochs = 5

# # Load Data

GTZAN = GtzanData()
print('Dataset created:')
print('train_x shape: %s' % (str(GTZAN.train_x.shape)))
print('train_y shape: %s' % (str(GTZAN.train_y.shape)))
print('test_x shape: %s' % (str(GTZAN.test_x.shape)))
print('test_y shape: %s' % (str(GTZAN.test_y.shape)))
print('get firstitem: ', str(GTZAN[0]))
print(len(GTZAN))

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

train_dataset = GtzanDataset(GTZAN.train_x, GTZAN.train_y, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

class Trainer:  
  def __init__(self, config):
    self.name = config.name
    self.model_type = config.model_type
    print('Training of ', self.name, ' just started...')
    # parameters
    self.level_of_dropout = config.level_of_dropout
    self.pooling_type = config.pooling_type
    self.activation_fn = config.activation_fn
    self.epochs = config.epochs
    self.batch_size = config.batch_size
    self.display_logs = config.display_logs
    self.use_tensorboard = config.use_tensorboard

    self.cuda = True if torch.cuda.is_available() else False
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.log_interval = 100
    self.globaliter = 0

    self.training_loss_values = []
    self.test_accuracies = []
    self.final_avg_loss = None
    self.final_accuracy = None
    
    torch.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

    self.train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
        batch_size=self.batch_size, shuffle=True, **kwargs)

    self.test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])),
        batch_size=1000, shuffle=True, **kwargs)

    self.model = CNN(
        self.name,
        level_of_dropout=self.level_of_dropout,
        pooling_type=self.pooling_type,
        activation_fn=self.activation_fn,
        use_tensorboard=self.use_tensorboard
    ).to(self.device)
    
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
   
  
  def train(self, epoch):

    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      
      self.globaliter += 1
      data, target = data.to(self.device), target.to(self.device)

      self.optimizer.zero_grad()
      predictions = self.model(data)

      loss = F.nll_loss(predictions, target)
      loss.backward()
      self.optimizer.step()

      if batch_idx % self.log_interval == 0:
        if self.display_logs:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(self.train_loader.dataset),
                  100. * batch_idx / len(self.train_loader), loss.item()))
        #Tensorboard logging at the end of each batch
        if self.use_tensorboard:
            self.model.train_summary_writer.add_scalar("loss", loss.item(), self.globaliter)
        self.training_loss_values.append(loss.item())

            
  def test(self, epoch):
    self.model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0

    with torch.no_grad():
      for data, target in self.test_loader:
        data, target = data.to(self.device), target.to(self.device)
        predictions = self.model(data)

        test_loss += F.nll_loss(predictions, target, reduction='sum').item()
        prediction = predictions.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

      test_loss /= len(self.test_loader.dataset)
      accuracy = 100. * correct / len(self.test_loader.dataset)
      if self.display_logs:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(self.test_loader.dataset), accuracy))
      
      #Tensorboard logging, but at the end of each testing epoch
      if self.use_tensorboard:
        self.model.test_summary_writer.add_scalar("loss", test_loss, self.globaliter)
        self.model.test_summary_writer.add_scalar("accuracy", accuracy, self.globaliter)
      self.test_accuracies.append(accuracy)
    self.final_avg_loss = test_loss
    self.final_accuracy = accuracy
  def record_performance_data(self):
    for epoch in range(1, self.epochs + 1):
      self.train(epoch)
      self.test(epoch)
    if self.use_tensorboard:
      self.model.train_summary_writer.close()
      self.model.test_summary_writer.close()
    print('Training of ',                   self.name,
          ' completed with average loss: ', self.final_avg_loss,
          ' and accuracy: ',                self.final_accuracy)

    return self.final_avg_loss, self.final_accuracy
  
# train_loader = DataLoader(dataset=np.append(X_train, y_train, axis=1), batch_size=batch_size)
# # test_dataset = datasets.MNIST(
# #     root="dataset/", train=False, transform=transforms.ToTensor(), download=True
# # )
# test_loader = DataLoader(dataset=np.append(X_test, y_test, axis=1), batch_size=batch_size, shuffle=True)

# # Initialize network
# model = CNN().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train Network
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # Get data to cuda if possible
#         data = data.to(device=device)
#         targets = targets.to(device=device)

#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)

#         # backward
#         optimizer.zero_grad()
#         loss.backward()

#         # gradient descent or adam step
#         optimizer.step()

# # Check accuracy on training & test to see how good our model


# def check_accuracy(loader, model):
#     if loader.dataset.train:
#         print("Checking accuracy on training data")
#     else:
#         print("Checking accuracy on test data")

#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)

#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#         print(
#             f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
#         )

#     model.train()


# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)