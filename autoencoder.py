import os
import pickle
from copy import deepcopy
import random

import torch
import torch.nn as nn

from autoencoder_model import BasicAutoencoder, PolicyAutoencoder, SmallerPolicyAutoencoder
from compute_transform import create_transform, create_transform_for_policy
from env import make_state
from env_render import render_state_tensor, render_state_tensor_for_policy
from utils import Ball, Vec2



def render_random_batch(batch_size, state_builder, transform=None, use_policy_render=False):
  states = [state_builder() for _ in range(batch_size)]
  if use_policy_render:
    return render_state_tensor_for_policy(states, transform=transform)
  return render_state_tensor(states, transform=transform)


def train(config, model):
  device = config['device']
  use_wandb = config['use_wandb']
  use_policy_render = config['use_policy_render']
  batch_size = config['batch_size']
  iters = config['iters']
  model_type = config['model_type']
  lr = config['lr']
  run_name = config['run_name']
  state_builder = config['state_builder']

  model = model.to(device)
  # TODO: figure out how to serialize when using torch.compile
  compiled_model = torch.compile(model, mode="reduce-overhead")

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  if use_policy_render:
    transform = create_transform_for_policy()
  else:
    transform = create_transform()

  if use_wandb:
    import wandb
    # override non string values
    wandb_config = {**deepcopy(config), 'device': str(device), 'model_type': model_type.__class__.__name__}
    # remove non serializable values
    wandb_config.pop('state_builder')
    wandb.init(project='minigolf_es_autoencoder', config=config, name=run_name)

  for i in range(iters):
    inputs = render_random_batch(batch_size, state_builder, transform=transform, use_policy_render=use_policy_render).to(device)
    outputs = compiled_model(inputs)


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
  # remove the state_builder since it's not serializable
  config.pop('state_builder')
  with open(os.path.join(run_name, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)


def default_config():
  return {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_wandb': True,
    'use_policy_render': False,
    'batch_size': 64,
    'iters': 1000,
    'model_type': BasicAutoencoder,
    'lr': 1e-3,
    'run_name': 'basic_autoencoder',
    'state_builder': make_state,
  }

def build_state():
  s = make_state()
  ball_start = Vec2(random.random(), random.random()) * 256
  s['ball'] = Ball(ball_start)
  s['ball_start'] = ball_start
  return s

if __name__ == '__main__':
  config = default_config()
  config['model_type'] = SmallerPolicyAutoencoder
  config['use_policy_render'] = True
  constructor_args = {'out_channels': 16, 'final_channel_factor': 4}
  config['constructor_args'] = constructor_args
  model = config['model_type'](**constructor_args)

  # # temp: load pretrained model
  # model.load_state_dict(torch.load('policy_autoencoder_smaller_1/model_final.pt'))


  config['iters'] = 3000
  # config['lr'] = 5e-4
  config['batch_size'] = 128 
  config['state_builder'] = build_state
  config['run_name'] = 'policy_autoencoder_smaller_2'
  train(config, model)
