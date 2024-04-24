import numpy as np
import pygame
from env import make_state
from env_render import render_state

import torch
# transforms
from torchvision.transforms import Normalize


def create_transform():
  """Create a virtual batch normalization layer."""

  # Create a batch of states
  states = [make_state() for _ in range(64)]

  # Render all states
  surfaces = [render_state(state) for state in states]

  # Convert the surfaces to tensors
  surface_rgb = [
      np.ascontiguousarray(pygame.surfarray.pixels3d(surface))
      for surface in surfaces
  ]  # list of numpy arrays of shape (size, size, 3)

  # Convert the RGB arrays to a PyTorch tensor
  # first convert the list of numpy arrays to a numpy array of shape (64, size, size, 3)
  surface_tensor = torch.from_numpy(np.stack(surface_rgb)).float()

  # compute mean and std per channel
  mean = surface_tensor.mean(dim=(0, 1, 2))
  std = surface_tensor.std(dim=(0, 1, 2))

  # Create a transform to normalize the pixel values
  transform = Normalize(mean, std)

  return transform
