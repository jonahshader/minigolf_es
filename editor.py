import pygame
from enum import Enum
from env import make_state, save_state
from env_render import render_state
from utils import Vec2, Wall, Line, Ball, Hole, CourseSurface, Rect


class Tool(Enum):
  WALL = 0
  BALL = 1
  HOLE = 2
  SAND = 3


if __name__ == '__main__':
  print('Testing state rendering...')

  def state_builder():
    return make_state(wall_chance=0)

  state = state_builder()
  screen = pygame.display.set_mode(
      (state['size'], state['size']), pygame.SCALED | pygame.RESIZABLE)

  clock = pygame.time.Clock()

  running = True
  mouse_pressed = False

  current_tool = Tool.WALL

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False
        elif event.key == pygame.K_SPACE:
          state = make_state()
        elif event.key == pygame.K_z:
          if current_tool == Tool.WALL and len(state['walls']) > 0:
            state['walls'].pop()
          elif current_tool == Tool.SAND and len(state['surfaces']) > 0:
            state['surfaces'].pop()
        elif event.key == pygame.K_w:
          current_tool = Tool.WALL
        elif event.key == pygame.K_b:
          current_tool = Tool.BALL
        elif event.key == pygame.K_h:
          current_tool = Tool.HOLE
        elif event.key == pygame.K_d:
          current_tool = Tool.SAND
        elif event.key == pygame.K_s:
          save_state(state)
          state = state_builder()

      elif event.type == pygame.MOUSEBUTTONDOWN:
        mouse_pressed = True
        x, y = pygame.mouse.get_pos()
        if current_tool == Tool.WALL:
          state['walls'].append(Wall(Line(Vec2(x, y), Vec2(x, y))))
        elif current_tool == Tool.BALL:
          state['ball'] = Ball(Vec2(x, y))
        elif current_tool == Tool.HOLE:
          state['hole'] = Hole(Vec2(x, y))
        elif current_tool == Tool.SAND:
          state['surfaces'].append(CourseSurface(Rect(Vec2(x, y), Vec2(0, 0))))

      elif event.type == pygame.MOUSEBUTTONUP:
        mouse_pressed = False

        if current_tool == Tool.SAND:
          # fix negative size cases
          for surface in state['surfaces']:
            if surface.rect.size.x < 0:
              surface.rect.pos.x += surface.rect.size.x
              surface.rect.size.x = abs(surface.rect.size.x)
            if surface.rect.size.y < 0:
              surface.rect.pos.y += surface.rect.size.y
              surface.rect.size.y = abs(surface.rect.size.y)

    if mouse_pressed:
      x, y = pygame.mouse.get_pos()
      if current_tool == Tool.WALL:
        state['walls'][-1].line.end = Vec2(x, y)
      elif current_tool == Tool.SAND:
        state['surfaces'][-1].rect.size = Vec2(x, y) - state['surfaces'][-1].rect.pos

    render_state(state, screen)
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
