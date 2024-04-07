import time
from model import BasicCNN
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch

from env_render import render_state

def policy_batched(screens, model, device='cpu'):
  # need to convert screen to tensor
  # grab the RGB array from the screen
  screen_rgb = [
      np.ascontiguousarray(pygame.surfarray.pixels3d(screen))
      for screen in screens
  ]

  # add a batch dimension to each screen
  screen_rgb = [np.expand_dims(screen, axis=0) for screen in screen_rgb]

  # concatenate the RGB arrays along the batch dimension
  screen_tensor = np.concatenate(screen_rgb, axis=0)

  # Convert the RGB array to a PyTorch tensor
  screen_tensor = torch.from_numpy(screen_tensor).float().to(device)

  # Normalize pixel values (divide by 255)
  screen_tensor = (screen_tensor / 255.0) * 2 - 1

  # Permute dimensions from BHWC to BCHW
  screen_tensor = screen_tensor.permute(0, 3, 1, 2)

  return model(screen_tensor)
  
if __name__ == '__main__':
  batch_size = 64
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = BasicCNN().to(device)
  
  states = [make_state() for _ in range(batch_size)]
  surfaces = [render_state(state) for state in states]

  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  with torch.no_grad():
    hit_directions = policy_batched(surfaces, model, device=device)
    start_time = time.time()
    start_event.record()
    hit_directions = policy_batched(surfaces, model, device=device)
    end_time = time.time()
    end_event.record()
    torch.cuda.synchronize()

  print(f'Inference time for batch size {batch_size}: {end_time - start_time:.4f} seconds')
  print(f'GPU time for batch size {batch_size}: {start_event.elapsed_time(end_event):.4f} ms')

  # for d in hit_directions:
  #   print(d)