import random
import pygame
from copy import deepcopy


class Vec2:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, other):
    return Vec2(self.x + other.x, self.y + other.y)

  def __sub__(self, other):
    return Vec2(self.x - other.x, self.y - other.y)

  def __mul__(self, scalar):
    return Vec2(self.x * scalar, self.y * scalar)

  def dot(self, other):
    return self.x * other.x + self.y * other.y

  def cross(self, other):
    return self.x * other.y - self.y * other.x

  def distance_to(self, other):
    return (self - other).dot(self - other) ** 0.5

  def magnitude(self):
    return self.dot(self) ** 0.5

  def set_magnitude(self, magnitude):
    return self * (magnitude / self.magnitude())

  def __truediv__(self, scalar):
    return Vec2(self.x / scalar, self.y / scalar)

  def __str__(self):
    return f"({self.x}, {self.y})"


class Circle:
  def __init__(self, center, radius):
    self.center = center
    self.radius = radius

  def contains(self, point):
    return (point - self.center).dot(point - self.center) <= self.radius * self.radius

  def render(self, surface, color=(0, 255, 0)):
    pygame.draw.circle(
        surface, color, (self.center.x, self.center.y), self.radius)


class Line:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __str__(self):
    return f"Line({self.start}, {self.end})"

  def intersect(self, other):
    p = self.start
    q = other.start
    r = self.end - self.start
    s = other.end - other.start

    r_cross_s = r.cross(s)
    q_minus_p = q - p

    if r_cross_s == 0:
      return None

    t = q_minus_p.cross(s) / r_cross_s
    u = q_minus_p.cross(r) / r_cross_s

    if 0 <= t <= 1 and 0 <= u <= 1:
      intersection = p + r * t
      return intersection
    return None


class Wall:
  thickness = 2

  def __init__(self, line, bounce_coeff=0.9):
    self.line = line
    self.bounce_coeff = bounce_coeff

  def render(self, surface):
    pygame.draw.line(surface, (0, 0, 0), (self.line.start.x, self.line.start.y),
                     (self.line.end.x, self.line.end.y), Wall.thickness)


class Rect:
  def __init__(self, pos, size):
    self.pos = pos
    self.size = size

  def contains(self, point):
    return self.pos.x <= point.x <= self.pos.x + self.size.x and self.pos.y <= point.y <= self.pos.y + self.size.y

  def create_random_inside(self):
    return Vec2(self.pos.x + random.random() * self.size.x, self.pos.y + random.random() * self.size.y)


class Ball:
  friction = 3  # 1/s^2
  radius = 3

  def __init__(self, pos):
    self.pos = pos
    self.vel = Vec2(0, 0)

  def render(self, surface):
    pygame.draw.circle(surface, (0, 0, 0),
                       (self.pos.x, self.pos.y), Ball.radius)

  def update(self, state, dt) -> bool:
    """Update the ball's state and return whether the ball is stopped."""
    hole = state["hole"]
    walls = state["walls"]

    # move the ball
    old_pos = self.pos
    self.pos = self.pos + self.vel * dt

    # check if the ball hits a wall
    # simple collision resolution: move the ball back and reflect the velocity
    movement = Line(old_pos, self.pos)
    for wall in walls:
      intersection = wall.line.intersect(movement)
      if intersection is not None:
        self.pos = intersection
        normal = (wall.line.end - wall.line.start).set_magnitude(1)
        self.vel = self.vel - 2 * \
            self.vel.dot(normal) * normal * wall.bounce_coeff

    # check if the ball is in the hole
    if hole.contains(self.pos):
      self.vel = Vec2(0, 0)
      self.pos = deepcopy(hole.center)

    # reduce velocity due to friction
    vel_mag = self.vel.magnitude()
    new_vel_mag = max(0, vel_mag - Ball.friction * dt)
    self.vel = self.vel.set_magnitude(new_vel_mag)

    return new_vel_mag == 0
