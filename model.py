import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    self.pool1 = nn.AvgPool2d(2, 2) # (batch_size, 3, 128, 128)
    self.conv1 = nn.Conv2d(3, 8, 3, padding=1) # (batch_size, 8, 128, 128)
    self.pool2 = nn.MaxPool2d(2, 2) # (batch_size, 8, 64, 64)
    self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # (batch_size, 16, 64, 64)
    self.pool3 = nn.MaxPool2d(4, 4) # (batch_size, 16, 16, 16)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2) # (batch_size, 32, 8, 8)
    self.fc1 = nn.Linear(32 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 3)

  def forward(self, x):
    x = self.pool1(x)
    x = self.pool2(F.gelu(self.conv1(x)))
    x = self.pool3(F.gelu(self.conv2(x)))
    x = self.pool4(F.gelu(self.conv3(x)))
    x = torch.flatten(x, 1)
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))
    x = self.fc3(x)

    # we have 3 outputs: x, y, and magnitude.
    # group x y into a single vector, then multiply by sigmoid(magnitude).
    direction = x[:, 0:2]
    # normalize direction
    direction = direction / torch.linalg.vector_norm(direction, dim=1).unsqueeze(1)
    magnitude = torch.tanh(x[:, 2:3])
    return direction * magnitude # (batch_size, 2)

class ConstModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.direction = nn.Parameter(torch.randn(1, 2))
    self.magnitude = nn.Parameter(torch.randn(1))

  def forward(self, x):
    output = self.direction / torch.linalg.vector_norm(self.direction)
    output = output * torch.tanh(self.magnitude)
    # match the batch size
    return output.repeat(x.size(0), 1)