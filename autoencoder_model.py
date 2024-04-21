
import torch.nn as nn


class PolicyEncoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()

    act_fn = nn.ReLU()

    self.net = nn.Sequential(
      nn.Conv2d(2, out_channels, 3, padding=1),  # 256
      act_fn,
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2),  # 128
      act_fn,
      nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(2*out_channels, 3*out_channels,
                3, padding=1, stride=2),  # 64
      act_fn,
      nn.Conv2d(3*out_channels, 3*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(3*out_channels, 3*out_channels,
                3, padding=1, stride=2),  # 32
      act_fn,
      nn.Conv2d(3*out_channels, 3*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(3*out_channels, 4*out_channels,
                3, padding=1, stride=2),  # 16
      act_fn,
      nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
      act_fn,
      nn.Conv2d(4*out_channels, final_channel_factor*out_channels,
                3, padding=1, stride=2)  # 8
    )

  def forward(self, x):
    return self.net(x)
  
class PolicyDecoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()

    act_fn = nn.ReLU()

    self.net = nn.Sequential(
      # 8
      nn.ConvTranspose2d(final_channel_factor*out_channels, 4*out_channels, 3, padding=1),
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1,
                         stride=2, output_padding=1),  # 16
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 8*out_channels, 3, padding=1),  # 16
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1,
                         stride=2, output_padding=1),  # 32
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1),  # 32
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1,
                         stride=2, output_padding=1),  # 64
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1),  # 64
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1,
                         stride=2, output_padding=1),  # 128
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1),  # 128
      act_fn,
      nn.ConvTranspose2d(8*out_channels, 4*out_channels, 3, padding=1,
                         stride=2, output_padding=1),  # 256
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),  # 256
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 2, 3, padding=1),  # 256
    )

  def forward(self, x):
    return self.net(x)
  

class SmallerPolicyEncoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()
    act_fn = nn.ReLU()
    self.net = nn.Sequential(
      nn.Conv2d(2, out_channels, 3, padding=1, stride=2),  # 128
      act_fn,
      nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2),  # 64
      act_fn,
      nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2),  # 32
      act_fn,
      nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1, stride=2),  # 16
      act_fn,
      nn.Conv2d(4*out_channels, final_channel_factor*out_channels, 3, padding=1, stride=2)  # 8
    )

  def forward(self, x):
    return self.net(x)
  
class SmallerPolicyDecoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()
    act_fn = nn.ReLU()
    self.net = nn.Sequential(
      nn.ConvTranspose2d(final_channel_factor*out_channels, 4*out_channels, 3, padding=1, stride=2, output_padding=1),  # 16
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1, stride=2, output_padding=1),  # 32
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1, stride=2, output_padding=1),  # 64
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1, stride=2, output_padding=1),  # 128
      act_fn,
      nn.ConvTranspose2d(4*out_channels, 2, 3, padding=1, stride=2, output_padding=1),  # 256
    )

  def forward(self, x):
    return self.net(x)
  

class BasicCNNEncoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200):
    super().__init__()

    act_fn = nn.ReLU()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 256
        act_fn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2),  # 128
        act_fn,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(2*out_channels, 4*out_channels,
                  3, padding=1, stride=2),  # 64
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels,
                  3, padding=1, stride=2),  # 32
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels,
                  3, padding=1, stride=2),  # 16
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels,
                  3, padding=1, stride=2),  # 8
        nn.Flatten(),
        nn.Linear(8*8*4*out_channels, latent_dim) if latent_dim is not None else nn.Identity(),
    )

  def forward(self, x):
    return self.net(x)


class BasicCNNDecoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200):
    super().__init__()

    act_fn = nn.ReLU()

    self.net = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*8*8) if latent_dim is not None else nn.Identity(),
        nn.Unflatten(1, (4*out_channels, 8, 8)),

        # (8, 8)
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (16, 16)

        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (32, 32)

        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (64, 64)

        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (128, 128)
        act_fn,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (256, 256)
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    return self.net(x)
  
class PolicyAutoencoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()
    self.encoder = PolicyEncoder(out_channels=out_channels, final_channel_factor=final_channel_factor)
    self.decoder = PolicyDecoder(out_channels=out_channels, final_channel_factor=final_channel_factor)

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)
  
class SmallerPolicyAutoencoder(nn.Module):
  def __init__(self, out_channels=16, final_channel_factor=1):
    super().__init__()
    self.encoder = SmallerPolicyEncoder(out_channels=out_channels, final_channel_factor=final_channel_factor)
    self.decoder = SmallerPolicyDecoder(out_channels=out_channels, final_channel_factor=final_channel_factor)

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)


class BasicAutoencoder(nn.Module):
  def __init__(self, latent_dim=200):
    super().__init__()
    self.encoder = BasicCNNEncoder(latent_dim=latent_dim)
    self.decoder = BasicCNNDecoder(latent_dim=latent_dim)

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)