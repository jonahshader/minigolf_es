import pygame


def render_state(state):
  pygame.init()

  ball = state['ball']
  hole = state['hole']
  walls = state['walls']
  size = state['size']

  # make render target
  surface = pygame.Surface((size, size))

  # clear surface with white
  surface.fill((255, 255, 255))

  # render game state
  hole.render(surface)
  for wall in walls:
    wall.render(surface)
  ball.render(surface)

  return surface
