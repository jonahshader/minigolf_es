from math import sqrt
from utils import Vec2, Line, Rect, Wall, Ball, Hole, CourseSurface
import random
import os
import pickle

  
def make_walls(ball_start, hole_start, size, wall_subsections, wall_chance, wall_overlap):
    ball_start = ball_start / size
    hole_start = hole_start / size
    walls = []
    for y in range(wall_subsections):
        for x in range(wall_subsections):
            region_x_start = x / wall_subsections
            region_y_start = y / wall_subsections
            region_x_end = (x + 1) / wall_subsections
            region_y_end = (y + 1) / wall_subsections
            region = Rect(Vec2(region_x_start, region_y_start), Vec2(region_x_end - region_x_start, region_y_end - region_y_start))
            if region.contains(ball_start) or region.contains(hole_start):
                continue
            if random.random() > wall_chance:
                continue

            region.pos -= Vec2(wall_overlap, wall_overlap) * region.size
            region.size += Vec2(wall_overlap, wall_overlap) * region.size * 2
            p1 = region.create_random_inside() * size
            p2 = region.create_random_inside() * size
            walls.append(Wall(Line(p1, p2)))
    return walls


def make_surfaces(size, min_surface_size, max_surface_size, num_surfaces):
    surfaces = []
    for _ in range(num_surfaces):
        surface_size = Vec2(random.random() * (max_surface_size - min_surface_size) + min_surface_size,
                            random.random() * (max_surface_size - min_surface_size) + min_surface_size)
        surface_pos = Vec2(random.random() * (size - surface_size.x),
                            random.random() * (size - surface_size.y))
        surface = CourseSurface(Rect(surface_pos, surface_size))
        surfaces.append(surface)
    return surfaces

def make_state(size=256, max_strokes=4, wall_subsections=5, wall_chance=0.5, wall_overlap=0.5, min_surface_size=32, max_surface_size=64, num_surfaces=5):
    # ball_start = Vec2(random.random() * 0.125 + 0.125, random.random() * 0.125 + 0.125) * size
    # hole_start = Vec2(size, size) - ball_start
    ball_start = Vec2(random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1) * size
    hole_start = Vec2(random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1) * size

    walls = make_walls(ball_start, hole_start, size,
                      wall_subsections, wall_chance, wall_overlap)
    
    surfaces = make_surfaces(size, min_surface_size, max_surface_size, num_surfaces)
    
    # add walls around the edge, inset
    inset = 3
    walls.append(Wall(Line(Vec2(inset, inset), Vec2(size-inset, inset))))
    walls.append(Wall(Line(Vec2(size-inset, inset), Vec2(size-inset, size-inset))))
    walls.append(Wall(Line(Vec2(size-inset, size-inset), Vec2(inset, size-inset))))
    walls.append(Wall(Line(Vec2(inset, size-inset), Vec2(inset, inset))))

    return {
        "ball": Ball(ball_start),
        "hole": Hole(hole_start),
        "walls": walls,
        "surfaces": surfaces,
        "strokes": 0,
        "size": size,
        "max_strokes": max_strokes
    }

def state_loss(state):
    ball = state["ball"]
    hole = state["hole"]
    strokes = state["strokes"]
    size = state["size"]

    stroke_loss = ball.pos.distance_to(hole.pos) / (size * sqrt(2))
    if hole.contains(ball.pos):
        stroke_loss = 0

    return max(stroke_loss + strokes - 1, 0)

def is_done(state):
    ball = state["ball"]
    hole = state["hole"]
    strokes = state["strokes"]
    max_strokes = state["max_strokes"]

    return hole.contains(ball.pos) or (ball.vel.magnitude() == 0 and strokes >= max_strokes)

def step(state, dt):
    ball = state["ball"]
    hole = state["hole"]
    strokes = state["strokes"]
    max_strokes = state["max_strokes"]

    ball_stopped = ball.update(state, dt)
    return ball_stopped and not hole.contains(ball.pos) and strokes < max_strokes

def run(state, dt):
    while not step(state, dt):
        if is_done(state):
            break

def act(state, hit_direction, max_speed=200):
    ball = state["ball"]
    ball.vel = hit_direction * max_speed
    state["strokes"] += 1

def save_state(state, directory='courses'):
  """Saves the state with the name 'state_{number}.pickle' where number is one greater than
  the largest number in the current directory."""
  if not os.path.exists(directory):
    os.makedirs(directory)
  files = os.listdir(directory)
  numbers = []
  for file in files:
    if file.startswith('state_') and file.endswith('.pickle'):
      try:
        number = int(file[len('state_'):-len('.pickle')])
        numbers.append(number)
      except:
        pass
  if len(numbers) == 0:
    number = 0
  else:
    number = max(numbers) + 1

  with open(f'{directory + "/state_"}{number}.pickle', 'wb') as f:
    pickle.dump(state, f)