import os
import pickle

from autoencoder import BasicAutoencoder, render_state_tensor, render_random_batch

import pygame
import torch
import numpy as np
from torchvision.transforms import Normalize

def render_autoencoder(model, state_builder, transform: Normalize):
  inputs = render_random_batch(1, state_builder, transform)
  with torch.no_grad():
    outputs = model(inputs)

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


if __name__ == '__main__':
  run_name = 'basic_1'

  # load the config and model
  with open(os.path.join(run_name, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)


  print(f'Loading model of type {config["model_type"].__name__}')
  model = config['model_type']()
  model.load_state_dict(torch.load(os.path.join(run_name, 'model_final.pt')))

  # load the transform
  with open(os.path.join(run_name, 'transform.pkl'), 'rb') as f:
    transform = pickle.load(f)

  # load the state builder
  state_builder = config['state_builder']

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
