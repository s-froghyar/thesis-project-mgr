import torch.nn.functional as F
import torch.nn as nn
import torch

class SegmentedCNN(nn.Module):
    def __init__(self):
        super(SegmentedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=(2, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(2, 1))

        self.fc1 = nn.Linear(64 * 32 * 18, 500)
        self.fc2 = nn.Linear(500, 10)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in')
        self.fc1.bias.data.fill_(0.01)

        self.pool = F.max_pool2d
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # self.imgs = []
        x = x.unsqueeze(1)
        # x = x.unsqueeze(0)
        # self.save_as_img(x)
        x = self.pool(self.conv1(x), 2)
        # self.save_as_img(x)
        x = F.relu(x)

        x = self.pool(self.conv2(x), 2)
        # self.save_as_img(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        # self.save_as_img(x)
        
        x = self.fc2(x)
        return x
