import torch.nn.functional as F
import torch.nn as nn

# Simple CNN
class AugerinoCNN(nn.Module):
    def __init__(self):
        super(AugerinoCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=(1, 3))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2)

        self.fc1 = nn.Linear(32*14*96, 10)
        self.pool = F.max_pool2d
        
    def forward(self, x):
        x = x.unsqueeze(1)
        print(x.shape)
        x = F.relu(self.pool(self.conv1(x), 2))
        print(x.shape)
        x = F.relu(self.pool(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return F.softmax(x, dim=1)
