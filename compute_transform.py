import numpy as np
import pygame
from env import make_state
from env_render import render_state, render_state_for_policy

import torch
# transforms
from torchvision.transforms import Normalize

def create_transform(states=None):
  # Create a batch of states
  if states is None:
    states = [make_state() for _ in range(512)]

  # Render all states
  surfaces = [render_state(state) for state in states]

  # Convert the surfaces to tensors
  surface_rgb = [
      np.ascontiguousarray(pygame.surfarray.pixels3d(surface))
      for surface in surfaces
  ] # list of numpy arrays of shape (size, size, 3)

  # Convert the RGB arrays to a PyTorch tensor
  # first convert the list of numpy arrays to a numpy array of shape (512, size, size, 3)
  surface_tensor = torch.from_numpy(np.stack(surface_rgb)).float()
  

  # compute mean and std per channel
  mean = surface_tensor.mean(dim=(0, 1, 2))
  std = surface_tensor.std(dim=(0, 1, 2))

  # Create a transform to normalize the pixel values
  transform = Normalize(mean, std)

  return transform

def create_transform_for_policy(states=None):
  # this function is similar to create_transform, but it uses render_state_for_policy
  # instead of render_state
  if states is None:
    states = [make_state() for _ in range(512)]

  wall_surface = pygame.Surface((states[0]['size'], states[0]['size']))
  hole_surface = pygame.Surface((states[0]['size'], states[0]['size']))

  # Convert the surfaces to tensors
  surface_rgb = []
  for state in states:
    render_state_for_policy(state, wall_surface, None, hole_surface)
    wall_np = np.ascontiguousarray(pygame.surfarray.pixels3d(wall_surface))
    hole_np = np.ascontiguousarray(pygame.surfarray.pixels3d(hole_surface))
    np_surface = wall_np + hole_np
    surface_rgb.append(np_surface)

  # Convert the RGB arrays to a PyTorch tensor
  # first convert the list of numpy arrays to a numpy array of shape (512, size, size, 3)
  surface_tensor = torch.from_numpy(np.stack(surface_rgb)).float()

  # compute mean and std per channel
  mean = surface_tensor.mean(dim=(0, 1, 2))
  std = surface_tensor.std(dim=(0, 1, 2))

  # Create a transform to normalize the pixel values
  transform = Normalize(mean, std)

  return transform