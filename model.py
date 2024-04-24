import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from autoencoder_res_model import BasicBlock


class BasicCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    self.pool1 = nn.AvgPool2d(2, 2)  # (batch_size, 3, 128, 128)
    self.conv1 = nn.Conv2d(3+2, 8, 3, padding=1)  # (batch_size, 8, 128, 128)
    self.pool2 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 64, 64)
    self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # (batch_size, 16, 64, 64)
    self.pool3 = nn.MaxPool2d(4, 4)  # (batch_size, 16, 16, 16)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2)  # (batch_size, 32, 8, 8)
    self.fc1 = nn.Linear(32 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 3)

    # CoordConv stuff
    pixel_x_pos = torch.linspace(-1, 1, 128).unsqueeze(0).expand(128, -1)
    pixel_y_pos = torch.linspace(-1, 1, 128).unsqueeze(1).expand(-1, 128)
    pixel_pos = torch.stack([pixel_x_pos, pixel_y_pos],
                            dim=0).unsqueeze(0)  # (1, 2, 128, 128)
    self.register_buffer('pixel_pos', pixel_pos)

  def forward(self, x):
    x = self.pool1(x)

    # x is (batch_size, 3, 128, 128)
    # want to add pixel position channels to get
    # (batch_size, 5, 128, 128)
    x = torch.cat([x, self.pixel_pos.expand(x.size(0), -1, -1, -1)], dim=1)

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
    direction = direction / \
        torch.linalg.vector_norm(direction, dim=1).unsqueeze(1)
    magnitude = torch.tanh(x[:, 2:3])
    return direction * magnitude  # (batch_size, 2)


class BasicCNNNoMag(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    self.pool1 = nn.AvgPool2d(2, 2)  # (batch_size, 3, 128, 128)
    self.conv1 = nn.Conv2d(3+2, 8, 3, padding=1)  # (batch_size, 8, 128, 128)
    self.pool2 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 64, 64)
    self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # (batch_size, 16, 64, 64)
    self.pool3 = nn.MaxPool2d(4, 4)  # (batch_size, 16, 16, 16)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2)  # (batch_size, 32, 8, 8)
    self.fc1 = nn.Linear(32 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 2)

    # CoordConv stuff
    pixel_x_pos = torch.linspace(-1, 1, 128).unsqueeze(0).expand(128, -1)
    pixel_y_pos = torch.linspace(-1, 1, 128).unsqueeze(1).expand(-1, 128)
    pixel_pos = torch.stack([pixel_x_pos, pixel_y_pos],
                            dim=0).unsqueeze(0)  # (1, 2, 128, 128)
    self.register_buffer('pixel_pos', pixel_pos)

  def forward(self, x):
    x = self.pool1(x)

    # x is (batch_size, 3, 128, 128)
    # want to add pixel position channels to get
    # (batch_size, 5, 128, 128)
    x = torch.cat([x, self.pixel_pos.expand(x.size(0), -1, -1, -1)], dim=1)

    x = self.pool2(F.gelu(self.conv1(x)))
    x = self.pool3(F.gelu(self.conv2(x)))
    x = self.pool4(F.gelu(self.conv3(x)))
    x = torch.flatten(x, 1)
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))
    x = self.fc3(x)

    # we get the magnitude from the norm of the direction
    magnitude = torch.linalg.vector_norm(x, dim=1).unsqueeze(1)
    # normalize direction then scale by sigmoid(magnitude)
    return x * F.tanh(magnitude) / magnitude


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


class TinyCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    self.pool1 = nn.AvgPool2d(4, 4)  # (batch_size, 3, 64, 64)
    # (batch_size, 8, 32, 32)
    self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
    self.pool2 = nn.MaxPool2d(4, 4)  # (batch_size, 8, 8, 8)
    # (batch_size, 8, 4, 4)
    self.conv2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
    self.fc1 = nn.Linear(8 * 4 * 4, 16)
    self.fc2 = nn.Linear(16, 16)
    self.fc3 = nn.Linear(16, 2)

  def forward(self, x):
    x = self.pool1(x)
    x = self.pool2(F.gelu(self.conv1(x)))
    x = F.gelu(self.conv2(x))
    x = torch.flatten(x, 1)
    x = F.gelu(self.fc1(x))
    x = x + F.gelu(self.fc2(x))
    x = F.tanh(self.fc3(x))
    return x


class TinyCNN2(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    features = 8
    self.pool1 = nn.AvgPool2d(2, 2)  # (batch_size, 3, 128, 128)
    # (batch_size, 8, 128, 128)
    self.conv1 = nn.Conv2d(3+2, features, 3, padding=1)
    # (batch_size, 8, 128, 128)
    self.conv2 = nn.Conv2d(features, features, 3, padding=1)
    self.pool2 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 64, 64)
    # (batch_size, 8, 64, 64)
    self.conv3 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 64, 64)
    self.conv4 = nn.Conv2d(features, features, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 32, 32)
    # (batch_size, 8, 32, 32)
    self.conv5 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 32, 32)
    self.conv6 = nn.Conv2d(features, features, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 16, 16)
    # (batch_size, 8, 16, 16)
    self.conv7 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 16, 16)
    self.conv8 = nn.Conv2d(features, 2, 3, padding=1)
    self.pool5 = nn.AvgPool2d(16, 16)  # (batch_size, 2, 1, 1)

    # CoordConv stuff
    pixel_x_pos = torch.linspace(-1, 1, 128).unsqueeze(0).expand(128, -1)
    pixel_y_pos = torch.linspace(-1, 1, 128).unsqueeze(1).expand(-1, 128)
    pixel_pos = torch.stack([pixel_x_pos, pixel_y_pos], dim=0).unsqueeze(0)
    self.register_buffer('pixel_pos', pixel_pos)

  def forward(self, x):
    x = self.pool1(x)  # (batch_size, 3, 128, 128)

    # want to add pixel position channels to get
    # (batch_size, 5, 128, 128)
    x = torch.cat([x, self.pixel_pos.expand(x.size(0), -1, -1, -1)], dim=1)
    # each conv pair is a residual block
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x)) + x
    x = self.pool2(x)
    x = F.gelu(self.conv3(x)) + x
    x = F.gelu(self.conv4(x)) + x
    x = self.pool3(x)
    x = F.gelu(self.conv5(x)) + x
    x = F.gelu(self.conv6(x)) + x
    x = self.pool4(x)
    x = F.gelu(self.conv7(x)) + x
    x = F.gelu(self.conv8(x))
    x = self.pool5(x)
    # tanh to force the output to be in the range [-1, 1]
    # reshape to (batch_size, 2)
    return F.tanh(x.view(x.size(0), 2))


class TinyCNN3(nn.Module):
  def __init__(self):
    super().__init__()
    # (batch_size, 3, 256, 256)
    features = 16
    self.pool1 = nn.AvgPool2d(2, 2)  # (batch_size, 3, 128, 128)
    # (batch_size, 8, 128, 128)
    self.conv1 = nn.Conv2d(3+2, features, 5, padding=2)
    # (batch_size, 8, 128, 128)
    self.conv2 = nn.Conv2d(features, features, 3, padding=1)
    self.pool2 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 64, 64)
    # (batch_size, 8, 64, 64)
    self.conv3 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 64, 64)
    self.conv4 = nn.Conv2d(features, features, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 32, 32)
    # (batch_size, 8, 32, 32)
    self.conv5 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 32, 32)
    self.conv6 = nn.Conv2d(features, features, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2)  # (batch_size, 8, 16, 16)
    # (batch_size, 8, 16, 16)
    self.conv7 = nn.Conv2d(features, features, 3, padding=1)
    # (batch_size, 8, 16, 16)
    self.conv8 = nn.Conv2d(features, 2, 3, padding=1)
    self.pool5 = nn.AvgPool2d(16, 16)  # (batch_size, 2, 1, 1)

    # CoordConv stuff
    pixel_x_pos = torch.linspace(-1, 1, 128).unsqueeze(0).expand(128, -1)
    pixel_y_pos = torch.linspace(-1, 1, 128).unsqueeze(1).expand(-1, 128)
    pixel_pos = torch.stack([pixel_x_pos, pixel_y_pos], dim=0).unsqueeze(0)
    self.register_buffer('pixel_pos', pixel_pos)

  def forward(self, x):
    x = self.pool1(x)  # (batch_size, 3, 128, 128)

    # want to add pixel position channels to get
    # (batch_size, 5, 128, 128)
    x = torch.cat([x, self.pixel_pos.expand(x.size(0), -1, -1, -1)], dim=1)
    # each conv pair is a residual block
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x)) + x
    x = self.pool2(x)
    x = F.gelu(self.conv3(x)) + x
    x = F.gelu(self.conv4(x)) + x
    x = self.pool3(x)
    x = F.gelu(self.conv5(x)) + x
    x = F.gelu(self.conv6(x)) + x
    x = self.pool4(x)
    x = F.gelu(self.conv7(x)) + x
    x = F.gelu(self.conv8(x))
    x = self.pool5(x)
    # tanh to force the output to be in the range [-1, 1]
    # reshape to (batch_size, 2)
    return F.tanh(x.view(x.size(0), 2))


class ResModel1(nn.Module):
  def __init__(self, in_channels=32, in_width=16, projection_channels=4, levels=3):
    super().__init__()
    current_channels = projection_channels
    current_width = in_width
    layers = []
    # TODO: maybe try breaking the projection into multiple layers
    layers.append(nn.Conv2d(in_channels, projection_channels, 1))

    for _ in range(levels):
      layers.append(BasicBlock(current_channels, F.relu, scaling=True))
      current_channels *= 2
      current_width //= 2

    layers.append(nn.Conv2d(current_channels, 2, current_width))
    layers.append(nn.Flatten())
    layers.append(nn.Tanh())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
