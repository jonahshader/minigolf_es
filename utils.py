import random

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
  def __init__(self, line, bounce_coeff=0.9):
    self.line = line
    self.bounce_coeff = bounce_coeff

class Rect:
  def __init__(self, pos, size):
    self.pos = pos
    self.size = size

  def contains(self, point):
    return self.pos.x <= point.x <= self.pos.x + self.size.x and self.pos.y <= point.y <= self.pos.y + self.size.y
  
  def create_random_inside(self):
    return Vec2(self.pos.x + random.random() * self.size.x, self.pos.y + random.random() * self.size.y)

class Ball:
  def __init__(self, pos):
    self.pos = pos
    self.vel = Vec2(0, 0)

  def update(self, world, dt) -> bool:
    pass