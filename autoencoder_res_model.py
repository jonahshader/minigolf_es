import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
  """ResNet-ish block. Four configs possible:
  transpose=False, scaling=False : normal block
  transpose=False, scaling=True  : down block. doubles channels, halves width and height
  transpose=True, scaling=False  : normal transpose block
  transpose=True, scaling=True   : up block. halves channels, doubles width and height
  """
  def __init__(self, channels, act, transpose=False, scaling=False):
    super().__init__()
    self.act = act
    self.transpose = transpose
    self.scaling = scaling

    scale = 2 if scaling else 1

    self.bn1 = nn.BatchNorm2d(channels)
    self.bn2 = nn.BatchNorm2d(channels // scale if transpose else channels * scale)
    if transpose:
      out_channels = channels // scale
      self.conv1 = nn.ConvTranspose2d(channels, out_channels, 3, stride=scale, padding=1, output_padding=(1 if scaling else 0))
      self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
    else:
      out_channels = channels * scale
      self.conv1 = nn.Conv2d(channels, out_channels, 3, stride=scale, padding=1)
      self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

    if scaling:
      self.projection = nn.Conv2d(channels, out_channels, 1)
      if transpose:
        self.upscale = nn.Upsample(scale_factor=scale)
      else:
        self.downscale = nn.AvgPool2d(2, stride=2)

    # if scaling:
    #   if transpose:
    #     self.projection = nn.ConvTranspose2d(channels, out_channels, 2, stride=2, output_padding=1)
    #   else:
    #     self.projection = nn.Conv2d(channels, out_channels, 1)

  def forward(self, x):
    x_next = self.bn1(x)
    x_next = self.act(x_next)
    x_next = self.conv1(x_next)
    x_next = self.bn2(x_next)
    x_next = self.act(x_next)
    x_next = self.conv2(x_next)

    if self.scaling:
      if self.transpose:
        # project then upscale to save compute
        x_res = self.projection(x)
        x_res = self.upscale(x_res)
      else:
        # downscale then project to save compute
        x_res = self.downscale(x)
        x_res = self.projection(x_res)
    else:
      x_res = x

    # if self.scaling:
    #   x_res = self.projection(x)
    # else:
    #   x_res = x

    return x_next + x_res
  

class ResAutoencoder(nn.Module):
  def __init__(self, channels, block_pattern, act=F.relu):
    super().__init__()
    encoder_blocks = []
    current_channels = channels
    for scaling in block_pattern:
      encoder_blocks.append(BasicBlock(current_channels, act, transpose=False, scaling=scaling))
      if scaling:
        current_channels *= 2

    decoder_blocks = []
    for scaling in reversed(block_pattern):
      decoder_blocks.append(BasicBlock(current_channels, act, transpose=True, scaling=scaling))
      if scaling:
        current_channels //= 2

    self.encoder = nn.Sequential(*encoder_blocks)
    self.decoder = nn.Sequential(*decoder_blocks)

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)

  
