import math
import random
from env import make_state, is_done, step, act, state_loss
from utils import Vec2
import pygame

from env_render import render_state


def run():
  print('Testing random policy...')

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
      print(state_loss(state))
      state = make_state()

    if step(state, 1/60):
      # ball = state['ball']
      # hole = state['hole']
      # hit_direction = hole.pos - ball.pos
      # hit_direction = hit_direction.set_magnitude(1)
      test_hit_direction = Vec2(1, 0)
      act(state, test_hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()

if __name__ == '__main__':
  run()