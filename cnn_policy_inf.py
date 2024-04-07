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
    output = model(screen_tensor)
    # output is a tensor of shape (1, 3)
    # we want to pull out the scalars as direction, magnitude
    direction = output[0][0:2]
    magnitude = torch.sigmoid(output[0][2])
    hit_direction = (direction * magnitude).numpy().tolist()
    return Vec2(*hit_direction)


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
      # hit_angle = random.uniform(0, 2 * math.pi)
      # hit_speed = math.tanh(random.random() * 10 - 5) * 400
      # hit_direction = Vec2(math.cos(hit_angle) * hit_speed, math.sin(hit_angle) * hit_speed)

      # ball = state['ball']
      # hole = state['hole']
      # hit_direction = hole.pos - ball.pos
      # hit_direction.set_magnitude(hit_direction.magnitude() * 2)

      hit_direction = policy(surface, model)
      print(f'hit_direction: {hit_direction}')
      act(state, hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(165)

  pygame.quit()


if __name__ == '__main__':
  run(BasicCNN())
