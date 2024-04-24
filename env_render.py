import numpy as np
import pygame
import torch
from env import make_state
from utils import Vec2


def render_state(state, surface=None, extras=False, no_clear=False):
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
  if not no_clear:
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


def render_state_for_policy(state, wall_surface=None, ball_surface=None, hole_surface=None):
  pygame.init()

  ball = state['ball']
  hole = state['hole']
  walls = state['walls']
  size = state['size']

  offset = ball.pos - Vec2(size / 2, size / 2)

  scale = 0.5

  # render to individual surfaces
  if wall_surface is not None:
    wall_surface.fill((0, 0, 0))
    for wall in walls:
      wall.render_for_policy(wall_surface, offset, size, scale)

  if ball_surface is not None:
    ball_surface.fill((0, 0, 0))
    ball.render_for_policy(ball_surface, offset, size, scale)

  if hole_surface is not None:
    hole_surface.fill((0, 0, 0))
    hole.render_for_policy(hole_surface, offset, size, scale)


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


def render_state_tensor_for_policy(states, transform=None):
  wall_surface = pygame.Surface((states[0]['size'], states[0]['size']))
  hole_surface = pygame.Surface((states[0]['size'], states[0]['size']))
  # skip ball since it is always in the center of the screen.

  tensors = []
  for state in states:
    render_state_for_policy(state, wall_surface, None, hole_surface)
    wall_np = np.ascontiguousarray(pygame.surfarray.pixels3d(wall_surface))
    hole_np = np.ascontiguousarray(pygame.surfarray.pixels3d(hole_surface))
    # drop the third channel (ball)
    wall_np = wall_np[:, :, :2]
    hole_np = hole_np[:, :, :2]

    np_surface = wall_np + hole_np
    surface_rgb = torch.from_numpy(np_surface).float()
    surface_rgb = surface_rgb.permute(2, 0, 1)
    if transform is not None:
      surface_rgb = transform(surface_rgb)
    surface_rgb = surface_rgb.unsqueeze(0)
    tensors.append(surface_rgb)

  return torch.cat(tensors, dim=0)


if __name__ == '__main__':
  print('Testing state rendering...')

  state_builder = make_state
  state = state_builder()
  screen = pygame.display.set_mode(
      (state['size'] * 2, state['size']), pygame.SCALED | pygame.RESIZABLE)

  wall_surface = pygame.Surface((state['size'], state['size']))
  ball_surface = pygame.Surface((state['size'], state['size']))
  hole_surface = pygame.Surface((state['size'], state['size']))

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
          state = state_builder()

    surface = render_state(state, screen)
    render_state_for_policy(state, wall_surface, ball_surface, hole_surface)
    # manually add these together using numpy
    # first, convert to numpy arrays
    wall_np = np.ascontiguousarray(pygame.surfarray.pixels3d(wall_surface))
    ball_np = np.ascontiguousarray(pygame.surfarray.pixels3d(ball_surface))
    hole_np = np.ascontiguousarray(pygame.surfarray.pixels3d(hole_surface))
    # add them together
    np_surface = wall_np + ball_np + hole_np
    # convert back to surface
    policy_surface = pygame.surfarray.make_surface(np_surface)
    screen.blit(policy_surface, (state['size'], 0))

    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
