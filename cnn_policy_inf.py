import math
import random
from model import BasicCNN
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch
import torch.nn.functional as F

from env_render import render_state

def policy(screen, model) -> tuple:
  # need to convert screen to tensor
  pixel_view = screen.get_view()
  pixel_view = np.array(pixel_view)
  screen_tensor = torch.from_numpy(pixel_view)
  # Normalize pixel values (divide by 255)
  screen_tensor = screen_tensor.float() / 255.0
  screen_tensor = screen_tensor * 2 - 1
  # Permute dimensions from HWC to CHW
  screen_tensor = screen_tensor.permute(2, 0, 1)
  # Add batch dimension
  screen_tensor = screen_tensor.unsqueeze(0)
  # Pass through the model
  with torch.no_grad():
    output = model(screen_tensor)
    # output is a tensor of shape (1, 3)
    # we want to pull out the scalars as direction, magnitude
    direction = output[0][0:2]
    magnitude = torch.sigmoid(output[0][2])
    # normalize x y then multiply with sigmoid(magnitude)
    return (direction * magnitude).item()


def run(model):
  print('Testing cnn policy...')

  state = make_state()
  screen = pygame.display.set_mode(
      (state['size'], state['size']), pygame.SCALED | pygame.RESIZABLE)

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
      # hit_angle = random.uniform(0, 2 * math.pi)
      # hit_speed = math.tanh(random.random() * 10 - 5) * 400
      # hit_direction = Vec2(math.cos(hit_angle) * hit_speed, math.sin(hit_angle) * hit_speed)

      ball = state['ball']
      hole = state['hole']
      hit_direction = hole.pos - ball.pos
      hit_direction.set_magnitude(hit_direction.magnitude() * 2)

      act(state, hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(165)

  pygame.quit()

if __name__ == '__main__':
  run(BasicCNN())