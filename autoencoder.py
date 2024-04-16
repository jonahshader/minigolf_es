import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNNEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    in_channels = 3
    out_channels = 16
    latent_dim = 64
    act_fn = nn.ReLU()

    self.net = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1), # 128
      act_fn,
      nn.Conv2d(out_channels, out_channels, 3, padding=1), 
      act_fn,
      nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # 64
      act_fn,
      nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # 32
      act_fn,
      nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(4*out_channels, 8*out_channels, 3, padding=1, stride=2), # 16
      act_fn,
      nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(8*out_channels, 16*out_channels, 3, padding=1, stride=2), # 8
      act_fn,
      nn.Flatten(),
      nn.Linear(8*8*16*out_channels, latent_dim),
    )

  def forward(self, x):
    return self.net(x)
  
class BasicCNNDecoder(nn.Module):
  def __init__(self):
    super().__init__()

    in_channels = 16
    out_channels = 8
    act_fn = nn.ReLU()

    self.net = nn.Sequential(
      nn.Linear(in_channels, 8*8*16*out_channels),
      act_fn,
      nn.Unflatten(1, (16*out_channels, 8, 8)),
      nn.ConvTranspose2d(16*out_channels, 8*out_channels, 3, padding=1, stride=2, output_padding=1), # 16
      act_fn,
      nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 4*out_channels, 3, padding=1, stride=2, output_padding=1), # 32
      act_fn,
      nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, stride=2, output_padding=1), # 64
      act_fn,
      nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
      act_fn,
      nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, stride=2, output_padding=1), # 128
      act_fn,
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(out_channels, 3, 3, padding=1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.net(x)
  
class BasicAutoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = BasicCNNEncoder()
    self.decoder = BasicCNNDecoder()

  def forward(self, x):
    return self.decoder(self.encoder(x))
  

# def train