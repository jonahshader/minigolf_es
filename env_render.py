import pygame
from env import make_state


def render_state(state, surface=None):
  pygame.init()

  ball = state['ball']
  hole = state['hole']
  walls = state['walls']
  surfaces = state['surfaces']
  size = state['size']

  # make render target if not provided
  if surface is None:
    surface = pygame.Surface((size, size))

  # clear surface with white
  surface.fill((64, 230, 12))

  # render surfaces
  for surf in surfaces:
    surf.render(surface)

  # render game state
  hole.render(surface)
  for wall in walls:
    wall.render(surface)
  ball.render(surface)

  return surface


if __name__ == '__main__':
  print('Testing state rendering...')

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

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
