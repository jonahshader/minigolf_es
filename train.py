import time
from model import BasicCNN
from env import make_state, is_done, step, act, run, state_loss
from utils import Vec2
import pygame
import numpy as np
import torch
# for parallel state running
import multiprocessing as mp

from env_render import render_state

def policy_batched(surfaces, model, device):
  # need to convert surface to tensor
  # grab the RGB array from the surface
  surface_rgb = [
      np.ascontiguousarray(pygame.surfarray.pixels3d(surface))
      for surface in surfaces
  ]

  # add a batch dimension to each surface
  surface_rgb = [np.expand_dims(surface, axis=0) for surface in surface_rgb]

  # concatenate the RGB arrays along the batch dimension
  surface_tensor = np.concatenate(surface_rgb, axis=0)

  # Convert the RGB array to a PyTorch tensor
  surface_tensor = torch.from_numpy(surface_tensor).float().to(device)

  # Normalize pixel values (divide by 255)
  surface_tensor = (surface_tensor / 255.0) * 2 - 1

  # Permute dimensions from BHWC to BCHW
  surface_tensor = surface_tensor.permute(0, 3, 1, 2)

  with torch.no_grad():
    return model(surface_tensor)
  
# closure for running a state
def run_state(state):
  run(state, 1/60)
  return state
  
def eval_batched(batch_size, model, device):
  # create a batch of states
  states = [make_state() for _ in range(batch_size)]
  
  # create a pool of processes for parallel state running
  pool = mp.Pool()

  # initial surfaces for all states is None
  surfaces = [None] * batch_size

  # eval the batch of states
  running = True
  while running:
    # render all states
    print("Rendering states")
    render_start = time.time()
    # TODO: parallelize this
    surfaces = [render_state(state, surface) for state, surface in zip(states, surfaces)]

    # compute hit directions for all states
    print("Computing hit directions")
    policy_start = time.time()
    with torch.no_grad():
      hit_directions = policy_batched(surfaces, model, device=device)
    hit_directions = [Vec2(*d.cpu().numpy().tolist()) for d in hit_directions]

    # act on all states
    print("Acting on states")
    act_start = time.time()
    for state, hit_direction in zip(states, hit_directions):
      if not is_done(state):
        act(state, hit_direction)

    # run all states until they are waiting for action
    print("Running states")
    run_start = time.time()
    # for state in states:
    #   run(state, 1/60)

    # parallelize the state running
    states = pool.map(run_state, states)

    # check if all states are done
    check_start = time.time()
    running = any(not is_done(state) for state in states)

    end_time = time.time()
    print(f"Rendering time: {policy_start - render_start:.4f} seconds")
    print(f"Policy time: {act_start - policy_start:.4f} seconds")
    print(f"Act time: {run_start - act_start:.4f} seconds")
    print(f"Run time: {check_start - run_start:.4f} seconds")
    print(f"Check time: {end_time - check_start:.4f} seconds")
    print(f"Total time: {end_time - render_start:.4f} seconds")

  # get the average loss for all states
  return sum([state_loss(state) for state in states]) / batch_size
  
if __name__ == '__main__':
  batch_size = 512
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

  # test the eval_batched function
  losses = eval_batched(batch_size, model, device)
  print("Losses:")
  print(losses)

  # for d in hit_directions:
  #   print(d)