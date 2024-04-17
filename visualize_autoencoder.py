import os
import pickle
import random

from autoencoder import BasicAutoencoder, render_state_tensor, render_random_batch
from env import make_state, step, act

import pygame
import torch
import numpy as np
from torchvision.transforms import Normalize

from utils import Vec2

def render_autoencoder(model, state_builder, transform: Normalize):
  inputs = render_random_batch(1, state_builder, transform)
  with torch.no_grad():
    latent = model.encoder(inputs)
    outputs = model.decoder(latent)

  # need to untransform the output
  outputs = outputs.squeeze(0)
  mean = transform.mean.clone().detach().unsqueeze(1).unsqueeze(2)
  std = transform.std.clone().detach().unsqueeze(1).unsqueeze(2)
  outputs = outputs * std + mean

  outputs = outputs.permute(1, 2, 0).cpu().numpy()
  outputs = np.clip(outputs, 0, 255).astype(np.uint8)
  outputs = pygame.surfarray.make_surface(outputs)

  # do the same for inputs
  inputs = inputs.squeeze(0)
  inputs = inputs * std + mean
  inputs = inputs.permute(1, 2, 0).cpu().numpy()
  inputs = np.clip(inputs, 0, 255).astype(np.uint8)
  inputs = pygame.surfarray.make_surface(inputs)
  return inputs, outputs


def test_random(model, state_builder, transform):
  # create pygame loop. wait for space press to render a new frame
  pygame.init()
  screen = pygame.display.set_mode((512, 256), pygame.SCALED | pygame.RESIZABLE)
  clock = pygame.time.Clock()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        inputs, outputs = render_autoencoder(model, state_builder, transform)
        screen.blit(inputs, (0, 0))
        screen.blit(outputs, (256, 0))
        pygame.display.flip()
        clock.tick(60)

def test_play(model, state_builder, transform):
  # create pygame loop. wait for space press to render a new frame
  pygame.init()
  screen = pygame.display.set_mode((512, 256), pygame.SCALED | pygame.RESIZABLE)
  clock = pygame.time.Clock()
  state = state_builder()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        state = state_builder()

    if step(state, 1/60):
      act(state, Vec2(random.random() * 2 - 1, random.random() * 2 - 1))
    inputs, outputs = render_autoencoder(model, lambda: state, transform)
    screen.blit(inputs, (0, 0))
    screen.blit(outputs, (256, 0))
    pygame.display.flip()
    clock.tick(60)


if __name__ == '__main__':
  run_name = 'basic_3_no_linear_longer'

  # load the config and model
  with open(os.path.join(run_name, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)

  model_type = config['model_type']
  constructor_args = config['constructor_args']
  state_builder = config['state_builder']

  print(f'Loading model of type {model_type.__name__}')
  model = model_type(**constructor_args)
  model.load_state_dict(torch.load(os.path.join(run_name, 'model_final.pt')))

  # load the transform
  with open(os.path.join(run_name, 'transform.pkl'), 'rb') as f:
    transform = pickle.load(f)

  test_play(model, state_builder, transform)

