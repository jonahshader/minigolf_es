from math import sqrt
from utils import Vec2, Line, Rect, Wall, Ball, Hole
import random


def make_walls(ball_start, hole_start, size, wall_subsections, wall_chance, wall_overlap):
  ball_start = ball_start / size
  hole_start = hole_start / size
  walls = []
  for y in range(wall_subsections):
    for x in range(wall_subsections):
      # create a region for this wall
      region_x_start = x / wall_subsections
      region_y_start = y / wall_subsections
      region_x_end = (x + 1) / wall_subsections
      region_y_end = (y + 1) / wall_subsections
      region = Rect(Vec2(region_x_start, region_y_start), Vec2(
          region_x_end - region_x_start, region_y_end - region_y_start))
      if region.contains(ball_start) or region.contains(hole_start):
        continue
      if random.random() > wall_chance:
        continue

      # expand region by wall_overlap proportion
      region.pos -= Vec2(wall_overlap, wall_overlap) * region.size
      region.size += Vec2(wall_overlap, wall_overlap) * region.size * 2

      # create a wall in this region
      p1 = region.create_random_inside() * size
      p2 = region.create_random_inside() * size

      walls.append(Wall(Line(p1, p2)))
  return walls


def make_state(size=256, max_strokes=4, wall_subsections=5, wall_chance=0.5, wall_overlap=0.5):
  ball_start = Vec2(random.random() * 0.125 + 0.125, random.random() * 0.125 + 0.125) * size
  hole_start = Vec2(size, size) - ball_start
  walls = make_walls(ball_start, hole_start, size,
                     wall_subsections, wall_chance, wall_overlap)
  
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
      "strokes": 0,
      "size": size,
      "max_strokes": max_strokes
  }


def state_loss(state):
  """Return the loss for the current state."""

  # rough scenarios in order from lowest to highest loss:
  # 1. ball is in hole after one stroke
  # 2. ball is in hole after more than one stroke
  # 3. ball is close to the hole after all strokes
  # 4. ball is far from the hole after all strokes
  # the following loss function is a heuristic that captures these scenarios

  ball = state["ball"]
  hole = state["hole"]
  strokes = state["strokes"]
  size = state["size"]

  stroke_loss = ball.pos.distance_to(hole.center) / (size * sqrt(2))
  if hole.contains(ball.pos):
    stroke_loss = 0

  return stroke_loss + strokes


def is_done(state):
  """Return whether the episode is done."""
  ball = state["ball"]
  hole = state["hole"]
  strokes = state["strokes"]
  max_strokes = state["max_strokes"]

  return hole.contains(ball.pos) or (ball.vel.magnitude() == 0 and strokes >= max_strokes)


def step(state, dt) -> bool:
  """Simulate the state for one step. Returns true if waiting for action.
  We are waiting for action if the ball is stopped and we aren't done yet."""

  ball = state["ball"]
  hole = state["hole"]
  strokes = state["strokes"]
  max_strokes = state["max_strokes"]

  # update ball
  ball_stopped = ball.update(state, dt)

  return ball_stopped and not hole.contains(ball.pos) and strokes < max_strokes


def run(state, dt):
  """Run the simulation until waiting for action."""
  while not step(state, dt):
    pass


def act(state, hit_direction):
  """Apply the hit direction to the ball."""
  ball = state["ball"]
  ball.vel = hit_direction
  state["strokes"] += 1
