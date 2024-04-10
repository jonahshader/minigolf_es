from model import BasicCNN
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch

from env_render import render_state


def policy(screen, model):
  # need to convert screen to tensor
  # grab the RGB array from the screen
  screen_rgb = np.ascontiguousarray(pygame.surfarray.pixels3d(screen))

  # Convert the RGB array to a PyTorch tensor
  screen_tensor = torch.from_numpy(screen_rgb)

  # Normalize pixel values (divide by 255)
  screen_tensor = (screen_tensor.float() / 255.0) * 2 - 1

  # Permute dimensions from HWC to CHW
  screen_tensor = screen_tensor.permute(2, 0, 1)

  # Add a batch dimension
  screen_tensor = screen_tensor.unsqueeze(0)

  with torch.no_grad():
    return model(screen_tensor)


def run(model):
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
      hit_direction = policy(surface, model)
      hit_direction = Vec2(*hit_direction[0].numpy().tolist())
      print(f'hit_direction: {hit_direction}')
      act(state, hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(165)

  pygame.quit()


if __name__ == '__main__':
  run(BasicCNN())
