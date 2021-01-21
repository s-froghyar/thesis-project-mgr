# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from tqdm import tqdm


from dataset import load_data, GtzanDataset

torchaudio.set_audio_backend("sox_io")

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
    def __init__(self, name, get_report_data=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=(1, 3))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2)

        self.fc1 = nn.Linear(32*14*96, 10)
        self.pool = F.max_pool2d
        
        if get_report_data:
          print('keeping log')
        
        print("\n\n-----------------CNN------------------\n\n")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool(self.conv1(x), 2))
        x = F.relu(self.pool(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return F.softmax(x, dim=1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.01
batch_size = 32
num_epochs = 5

is_test = True

# Load Data
GTZAN = load_data(is_test=is_test)

train_dataset = GtzanDataset(GTZAN.train_x, GTZAN.train_y, train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

test_dataset = GtzanDataset(GTZAN.test_x, GTZAN.test_y, train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN('yeet', use_tensorboard=False).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print('epoch num ', epoch)
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)