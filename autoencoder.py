import os
import pickle

import torch
import torch.nn as nn

from compute_transform import create_transform
from env import make_state
from env_render import render_state_tensor


class BasicCNNEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    in_channels = 3
    out_channels = 16
    latent_dim = 200
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
        nn.Conv2d(4*out_channels, 8*out_channels,
                  3, padding=1, stride=2),  # 32
        act_fn,
        nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(8*out_channels, 16*out_channels,
                  3, padding=1, stride=2),  # 16
        act_fn,
        nn.Conv2d(16*out_channels, 16*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(16*out_channels, 16*out_channels,
                  3, padding=1, stride=2),  # 8
        nn.Flatten(),
        nn.Linear(8*8*16*out_channels, latent_dim),
    )

  def forward(self, x):
    return self.net(x)


class BasicCNNDecoder(nn.Module):
  def __init__(self):
    super().__init__()

    latent_dim = 200
    in_channels = 3
    out_channels = 16

    act_fn = nn.ReLU()

    self.net = nn.Sequential(
        nn.Linear(latent_dim, 16*out_channels*8*8),
        nn.Unflatten(1, (16*out_channels, 8, 8)),

        # (8, 8)
        nn.ConvTranspose2d(16*out_channels, 16*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(16*out_channels, 16*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (16, 16)

        nn.ConvTranspose2d(16*out_channels, 16*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(16*out_channels, 8*out_channels, 3, padding=1,
                           stride=2, output_padding=1),  # (32, 32)

        nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(8*out_channels, 4*out_channels, 3, padding=1,
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


class BasicAutoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = BasicCNNEncoder()
    self.decoder = BasicCNNDecoder()

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)


def render_random_batch(batch_size, state_builder, transform=None):
  states = [state_builder() for _ in range(batch_size)]
  return render_state_tensor(states, transform=transform)


def train(config):
  device = config['device']
  use_wandb = config['use_wandb']
  batch_size = config['batch_size']
  iters = config['iters']
  model_type = config['model_type']
  lr = config['lr']
  run_name = config['run_name']
  state_builder = config['state_builder']

  model = model_type().to(device)

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  transform = create_transform()

  if use_wandb:
    import wandb
    # override non string values
    config = {**config, 'device': str(device), 'model_type': str(model_type)}
    # remove non serializable values
    config.pop('state_builder')
    wandb.init(project='minigolf_es_autoencoder', config=config, name=run_name)

  for i in range(iters):
    inputs = render_random_batch(batch_size, state_builder, transform).to(device)
    outputs = model(inputs)

    loss = criterion(outputs, inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
      print(f'Iter {i}, Loss: {loss.item()}')
      if use_wandb:
        wandb.log({'loss': loss.item(), 'iter': i})

  os.makedirs(run_name, exist_ok=True)
  # automatically add the run directory to .gitignore
  with open(".gitignore", "a") as f:
    f.write(f"\n/{run_name}")
  torch.save(model.state_dict(), os.path.join(run_name, 'model_final.pt'))
  with open(os.path.join(run_name, 'transform.pkl'), 'wb') as f:
    pickle.dump(transform, f)
  # also save the config
  with open(os.path.join(run_name, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)


def default_config():
  return {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_wandb': True,
    'batch_size': 64,
    'iters': 1000,
    'model_type': BasicAutoencoder,
    'lr': 1e-3,
    'run_name': 'basic_autoencoder',
    'state_builder': make_state,
  }


if __name__ == '__main__':
  config = default_config()
  config['run_name'] = 'basic_1'
  train(config)
