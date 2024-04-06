import pygame
import numpy as np

display_width, display_height = 1024, 1024

def initialize_pygame():
  pygame.init()
  internal_width, internal_height = 128, 128

  screen = pygame.display.set_mode((display_width, display_height))
  return screen, internal_width, internal_height


def render_state(state, screen, font=None, internal_width=128, internal_height=128, display=False):
  # Create a surface for rendering the internal resolution
  internal_surface = pygame.Surface((internal_width, internal_height))

  # Clear the internal surface
  internal_surface.fill((255, 255, 255))  # Fill with white color

  # Render the game state on the internal surface
  ball_x, ball_y = state["ball_position"]
  hole_x, hole_y = state["hole_position"]
  obstacles = state["obstacles"]

  pygame.draw.circle(internal_surface, (0, 0, 0),
                     (ball_x, ball_y), 10)  # Draw the ball
  pygame.draw.circle(internal_surface, (0, 255, 0),
                     (hole_x, hole_y), 15)  # Draw the hole

  # Draw obstacles
  for obstacle in obstacles:
    pygame.draw.rect(internal_surface, (255, 0, 0), obstacle)

  # Render text if a font is provided
  if font:
    text_surface = font.render("Score: 0", True, (0, 0, 0))
    internal_surface.blit(text_surface, (10, 10))

  # Scale the internal surface to the display resolution using nearest neighbor scaling
  scaled_surface = pygame.transform.scale(
      internal_surface, (display_width, display_height))

  # Blit the scaled surface onto the display
  screen.blit(scaled_surface, (0, 0))

  if display:
    # Update the display
    pygame.display.flip()


# Initialize pygame and create the surface once
screen, internal_width, internal_height = initialize_pygame()

# Load the font once (if needed)
font = pygame.font.Font(None, 36)

# Example usage
state1 = {"ball_position": (25, 50), "hole_position": (
    120, 100), "obstacles": [(75, 50, 12, 12)]}
render_state(state1, screen, font, internal_width,
             internal_height, display=True)

# Create a clock for limiting FPS
clock = pygame.time.Clock()
target_fps = 60

# Keep the window open until the user quits
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.MOUSEBUTTONDOWN:
      # Get the position of the mouse click
      mouse_pos = pygame.mouse.get_pos()
      mouse_pos = (mouse_pos[0] // 8, mouse_pos[1] // 8)
      print(f"Mouse down at {mouse_pos}")
    elif event.type == pygame.MOUSEBUTTONUP:
      # Get the position of the mouse click
      mouse_pos = pygame.mouse.get_pos()
      mouse_pos = (mouse_pos[0] // 8, mouse_pos[1] // 8)
      print(f"Mouse up at {mouse_pos}")

  render_state(state1, screen, font, internal_width,
               internal_height, display=True)
  clock.tick(target_fps)

# Clean up pygame
pygame.quit()
