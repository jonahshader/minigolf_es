import pygame
from env import make_state
import random

# Define grass colors
light_green = (122, 194, 111)
dark_green = (86, 157, 77)

def render_grass(size):
    # Create a surface for the grass texture
    grass_surface = pygame.Surface((size, size))

    # Draw the grassy texture
    for y in range(0, size, 2):
        for x in range(0, size, 2):
            color = light_green if random.random() > 0.5 else dark_green
            pygame.draw.rect(grass_surface, color, (x, y, 2, 2))

    return grass_surface

# Create the grass texture once
grass_texture = render_grass(256)  # Assuming size is 256, adjust if needed

def render_state(state, surface=None):
    # Check if pygame is initialized
    if not pygame.get_init():
        pygame.init()

    ball = state['ball']
    hole = state['hole']
    walls = state['walls']
    size = state['size']

    # Make render target if not provided
    if surface is None:
        surface = pygame.Surface((size, size))

    # Render grass texture
    surface.blit(grass_texture, (0, 0))

    # Render game state
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
