from model import BasicCNN, ConstModel, BasicCNNNoMag
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch
import pickle
import os

from compute_transform import create_transform

from env_render import render_state


def policy(screen, model, transform):
  # need to convert screen to tensor
  # grab the RGB array from the screen
  screen_rgb = np.ascontiguousarray(pygame.surfarray.pixels3d(screen))

  # Convert the RGB array to a PyTorch tensor
  screen_tensor = torch.from_numpy(screen_rgb).float()

  # permute dimensions from HWC to CHW
  screen_tensor = screen_tensor.permute(2, 0, 1)

  # apply the transformation to the tensor
  screen_tensor = transform(screen_tensor)

  # Add a batch dimension
  screen_tensor = screen_tensor.unsqueeze(0)

  with torch.no_grad():
    return model(screen_tensor)


def run(model, transform):
  print('Testing cnn policy...')

  state = make_state()
  screen = pygame.display.set_mode(
      (state['size'], state['size']), pygame.SCALED | pygame.RESIZABLE)
  surface = render_state(state, screen)

  clock = pygame.time.Clock()

  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False
        elif event.key == pygame.K_SPACE:
          state = make_state()

    if is_done(state):
      state = make_state()

    if step(state, 1/60):
      hit_direction = policy(surface, model, transform)
      hit_direction = Vec2(*hit_direction[0].numpy().tolist())
      print(f'hit_direction: {hit_direction}')
      act(state, hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(165)

  pygame.quit()


if __name__ == '__main__':
  run_name = 'testing_inference'
  model_type = BasicCNNNoMag

  model1 = model_type()
  model1.load_state_dict(torch.load(os.path.join(run_name, f'model_best.pt')))

  # try loading the transform
  try:
    with open(os.join(run_name, 'transform.pkl'), 'rb') as f:
      print('Loading transform...')
      transform = pickle.load(f)
  except:
    print('Transform not found, creating new one...')
    transform = create_transform()

  run(model1, transform)

