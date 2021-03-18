import torch.nn.functional as F
import torch.nn as nn

# Simple CNN
class SegmentedCNN(nn.Module):
    def __init__(self):
        super(SegmentedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=(1, 2), padding=(2, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=(1, 2), padding=(2, 1))

        self.fc1 = nn.Linear(64*64*4, 10)
        self.pool = F.max_pool2d
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool(self.conv1(x), 2))
        x = F.relu(self.pool(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
