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

  def __mul__(self, other):
    if type(other) == Vec2:
      return Vec2(self.x * other.x, self.y * other.y)
    return Vec2(self.x * other, self.y * other)

  def dot(self, other):
    return self.x * other.x + self.y * other.y

  def cross(self, other):
    return self.x * other.y - self.y * other.x

  def distance_to(self, other):
    return (self - other).dot(self - other) ** 0.5

  def magnitude(self):
    return self.dot(self) ** 0.5

  def set_magnitude(self, magnitude):
    if magnitude == 0:
      return Vec2(0, 0)
    return self * (magnitude / self.magnitude())

  def __truediv__(self, scalar):
    return Vec2(self.x / scalar, self.y / scalar)

  def __str__(self):
    return f"({self.x}, {self.y})"


class Hole:
  def __init__(self, pos, radius=4):
    self.pos = pos
    self.radius = radius + Ball.radius

  def contains(self, point):
    return (point - self.pos).dot(point - self.pos) <= self.radius * self.radius

  def render(self, surface):
    pygame.draw.circle(
        surface, (48, 172, 9), (self.pos.x, self.pos.y), self.radius - Ball.radius + 1)
    pygame.draw.circle(
        surface, (0, 0, 0), (self.pos.x, self.pos.y), self.radius - Ball.radius)

  def render_for_policy(self, surface, offset, size, scale):
    # render the hole, offset by the ball's position
    pygame.draw.circle(
        surface, (255, 0, 0), ((self.pos.x - size/2) * scale + size/2 - offset.x * scale, (self.pos.y - size/2) * scale + size/2 - offset.y * scale), self.radius - Ball.radius)


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

  def __init__(self, line, bounce_coeff=0.95):
    self.line = line
    self.bounce_coeff = bounce_coeff

  def render(self, surface):
    pygame.draw.line(surface, (48, 172, 9), (self.line.start.x-1, self.line.start.y+1),
                     (self.line.end.x-1, self.line.end.y+1), Wall.thickness)
    pygame.draw.line(surface, (200, 170, 150), (self.line.start.x, self.line.start.y),
                     (self.line.end.x, self.line.end.y), Wall.thickness)

  def render_for_policy(self, surface, offset, size, scale):
    # render the wall, offset by the ball's position
    pygame.draw.line(surface, (0, 255, 0), ((self.line.start.x - size/2) * scale + size/2 - offset.x * scale, (self.line.start.y - size/2) * scale + size/2 - offset.y * scale),
                     ((self.line.end.x - size/2) * scale + size/2 - offset.x * scale, (self.line.end.y - size/2) * scale + size/2 - offset.y * scale), Wall.thickness)


class Rect:
  def __init__(self, pos, size):
    self.pos = pos
    self.size = size

  def contains(self, point):
    return self.pos.x <= point.x <= self.pos.x + self.size.x and self.pos.y <= point.y <= self.pos.y + self.size.y

  def create_random_inside(self):
    return Vec2(self.pos.x + random.random() * self.size.x, self.pos.y + random.random() * self.size.y)


class Ball:
  friction = 25  # 1/s^2
  radius = 4

  def __init__(self, pos):
    self.pos = pos
    self.vel = Vec2(0, 0)

  def render(self, surface):
    pygame.draw.circle(surface, (48, 172, 9),
                       (self.pos.x-1, self.pos.y+1), Ball.radius)
    pygame.draw.circle(surface, (32, 115, 6),
                       (self.pos.x-1, self.pos.y+1), Ball.radius-1)
    pygame.draw.circle(surface, (225, 255, 255),
                       (self.pos.x, self.pos.y), Ball.radius)

  def render_for_policy(self, surface, offset, size, scale):
    # render the ball, offset by the ball's position
    pygame.draw.circle(surface, (0, 0, 255),
                       ((self.pos.x - size/2) * scale + size/2 - offset.x * scale, (self.pos.y - size/2) * scale + size/2 - offset.y * scale), Ball.radius)

  def update(self, state, dt) -> tuple[bool, bool]:
    """Update the ball's state and return whether the ball is stopped."""
    hole = state["hole"]
    walls = state["walls"]

    # move the ball
    old_pos = self.pos
    self.pos = self.pos + self.vel * dt

    # check if the ball hits a wall
    # simple collision resolution: move the ball back and reflect the velocity
    movement = Line(old_pos, self.pos)
    bounced = False
    for wall in walls:
      intersection = wall.line.intersect(movement)
      if intersection is not None:
        self.pos = old_pos
        normal = (wall.line.end - wall.line.start).set_magnitude(1)
        normal = Vec2(-normal.y, normal.x)
        self.vel = self.vel - normal * \
            wall.bounce_coeff * self.vel.dot(normal) * 2
        bounced = True

    # check if the ball is in the hole
    if hole.contains(self.pos):
      self.vel = Vec2(0, 0)
      self.pos = deepcopy(hole.pos)

    # reduce velocity due to friction
    vel_mag = self.vel.magnitude()
    new_vel_mag = max(0, vel_mag - Ball.friction * dt)
    self.vel = self.vel.set_magnitude(new_vel_mag)

    return new_vel_mag == 0, bounced
