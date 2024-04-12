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
      # hit_angle = random.uniform(0, 2 * math.pi)
      # hit_speed = math.tanh(random.random() * 10 - 5) * 400
      # hit_direction = Vec2(math.cos(hit_angle) * hit_speed, math.sin(hit_angle) * hit_speed)

      ball = state['ball']
      hole = state['hole']
      hit_direction = hole.pos - ball.pos
      # hit_direction = hit_direction.set_magnitude(hit_direction.magnitude() * (1/400))
      hit_direction = hit_direction.set_magnitude(1)
      act(state, hit_direction)

    surface = render_state(state, screen)
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()

if __name__ == '__main__':
  run()