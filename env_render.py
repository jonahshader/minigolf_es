import numpy as np
import pygame
import torch
from env import make_state
from utils import Vec2


def render_state(state, surface=None, extras=False):
  pygame.init()

  ball = state['ball']
  hole = state['hole']
  walls = state['walls']
  size = state['size']
  ball_start = state['ball_start']

  # make render target if not provided
  if surface is None:
    surface = pygame.Surface((size, size))

  # clear surface with white
  surface.fill((64, 230, 12))

  # render game state
  hole.render(surface)
  for wall in walls:
    wall.render(surface)

  if extras:
    # render ball start
    pygame.draw.circle(surface, (255, 0, 0),
                       (int(ball_start.x), int(ball_start.y)), 2)
  ball.render(surface)
  return surface


def render_state_for_policy(state, surface=None):
  pygame.init()

  ball = state['ball']
  hole = state['hole']
  walls = state['walls']
  size = state['size']

  offset = ball.pos - Vec2(size // 2, size // 2)

  if surface is None:
    surface = pygame.Surface((size, size))

  # clear surface with black
  surface.fill((0, 0, 0))

  scale = 0.5

  # render game state
  hole.render_for_policy(surface, offset, size, scale)
  for wall in walls:
    wall.render_for_policy(surface, offset, size, scale)

  ball.render_for_policy(surface, offset, size, scale)

  return surface

def render_state_tensor(states, transform=None):
  surface = None
  tensors = []
  for state in states:
    surface = render_state(state, surface)
    surface_rgb = np.ascontiguousarray(pygame.surfarray.pixels3d(surface))
    surface_rgb = torch.from_numpy(surface_rgb).float()
    surface_rgb = surface_rgb.permute(2, 0, 1)
    if transform is not None:
      surface_rgb = transform(surface_rgb)
    surface_rgb = surface_rgb.unsqueeze(0)
    tensors.append(surface_rgb)

  return torch.cat(tensors, dim=0)


if __name__ == '__main__':
  print('Testing state rendering...')

  state = make_state()
  screen = pygame.display.set_mode(
      (state['size'] * 2, state['size']), pygame.SCALED | pygame.RESIZABLE)

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

    surface = render_state(state, screen)
    policy_surface = render_state_for_policy(state)
    screen.blit(policy_surface, (state['size'], 0))
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
