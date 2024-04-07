import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    self.conv1 = nn.Conv2d(3, 8, 7, padding=3) # (batch_size, 8, 256, 256)
    self.pool1 = nn.MaxPool2d(2, 2) # (batch_size, 8, 128, 128)
    self.conv2 = nn.Conv2d(8, 16, 5, padding=2) # (batch_size, 16, 128, 128)
    self.pool2 = nn.MaxPool2d(4, 4) # (batch_size, 16, 32, 32)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2, 2) # (batch_size, 32, 16, 16)
    self.fc1 = nn.Linear(32 * 16 * 16, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 3)

  def forward(self, x):
    x = self.pool1(F.gelu(self.conv1(x)))
    x = self.pool2(F.gelu(self.conv2(x)))
    x = self.pool3(F.gelu(self.conv3(x)))
    x = torch.flatten(x, 1)
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))
    x = self.fc3(x)
    return x
