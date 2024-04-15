import os
import time
import signal
from model import BasicCNN, ConstModel, BasicCNNNoMag, TinyCNN, TinyCNN2
from env import make_state, is_done, step, act, run, state_loss
from utils import Vec2
import pygame
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from copy import deepcopy
# for parallel state running
import multiprocessing as mp
import pickle
import wandb

from env_render import render_state
from compute_transform import create_transform

print_timings = False

# create transform to normalize pixel values
# transform = create_transform()


def policy_batched(surfaces: list, model, device):
  # need to convert surface to tensor
  surface_tensors = []

  for surface in surfaces:
    # grab the RGB array from the surface
    surface_rgb = np.ascontiguousarray(pygame.surfarray.pixels3d(surface))
    # Convert the RGB array to a tensor
    surface_rgb = torch.from_numpy(surface_rgb).float()
    # permute dimensions from HWC to CHW
    surface_rgb = surface_rgb.permute(2, 0, 1)
    # Apply the transform
    surface_tensor = transform(surface_rgb)
    # Add a batch dimension to the tensor
    surface_tensor = surface_tensor.unsqueeze(0)
    # Append the tensor to the list
    surface_tensors.append(surface_tensor)

  # Concatenate the surface tensors along the batch dimension
  surface_tensor = torch.cat(surface_tensors, dim=0).to(device)

  with torch.no_grad():
    return model(surface_tensor)


# closure for running a state
def run_state(state):
  run(state, 1/60)
  return state


def eval_batched(states, model, device, pool, surfaces=None):
  # initial surfaces for all states is None
  if surfaces is None:
    surfaces = [None] * len(states)

  # eval the batch of states
  running = True
  while running:
    # render all states
    render_start = time.time()
    # TODO: parallelize this
    surfaces = [render_state(state, surface)
                for state, surface in zip(states, surfaces)]

    # compute hit directions for all states
    policy_start = time.time()
    with torch.no_grad():
      hit_directions = policy_batched(surfaces, model, device=device)
    hit_directions = [Vec2(*d.cpu().numpy().tolist()) for d in hit_directions]

    # act on all states
    act_start = time.time()
    for state, hit_direction in zip(states, hit_directions):
      if not is_done(state):
        act(state, hit_direction)

    # run all states until they are waiting for action
    run_start = time.time()
    # for state in states:
    #   run(state, 1/60)

    # parallelize the state running
    states = pool.map(run_state, states)

    # check if all states are done
    check_start = time.time()
    running = any(not is_done(state) for state in states)

    end_time = time.time()
    if print_timings:
      print(f"Rendering time: {policy_start - render_start:.4f} seconds")
      print(f"Policy time: {act_start - policy_start:.4f} seconds")
      print(f"Act time: {run_start - act_start:.4f} seconds")
      print(f"Run time: {check_start - run_start:.4f} seconds")
      print(f"Check time: {end_time - check_start:.4f} seconds")
      print(f"Total time: {end_time - render_start:.4f} seconds")

  # get the average loss for all states
  return [state_loss(state) for state in states], surfaces


def train_step(states, model, optimizer, batch_size, pool: mp.Pool, device='cpu', standard_dev=0.01, surface_batches=None):
  with torch.no_grad():
    # clone the states for each batch
    states_batches = [deepcopy(states) for _ in range(batch_size)]

    # clone the model for each batch
    models = [type(model)().to(device) for _ in range(batch_size)]
    # copy parameters from the original model to the cloned models
    for m in models:
      m.load_state_dict(model.state_dict())

    # create noise tensors matching the model parameters
    # only generate half, then apply each noise twice: once positive and once negative
    noises = [[torch.randn_like(p, device=device)
               for p in model.parameters()] for _ in range(batch_size // 2)]
    noises = noises + [[-noise for noise in noise_list]
                       for noise_list in noises]

    # add noise to the model parameters
    for m, noise_list in zip(models, noises):
      for p, noise in zip(m.parameters(), noise_list):
        p.add_(noise * standard_dev)

    # initial surfaces for all states is None
    if surface_batches is None:
      surface_batches = [[None] * len(states) for _ in range(batch_size)]

    # eval all models on batches of states
    running = True
    while running:
      # render all states
      if print_timings:
        print("Rendering")
      render_start = time.time()
      for state_batch, surface_batch in zip(states_batches, surface_batches):
        for i, (state, surface) in enumerate(zip(state_batch, surface_batch)):
          surface_batch[i] = render_state(state, surface)

      # compute hit directions for all states
      if print_timings:
        print("Policy")
      policy_start = time.time()
      hit_direction_batches = []
      for m, surface_batch in zip(models, surface_batches):
        hit_direction_batch = policy_batched(surface_batch, m, device=device)
        hit_direction_batch = [Vec2(*d.cpu().numpy().tolist())
                               for d in hit_direction_batch]
        hit_direction_batches.append(hit_direction_batch)

      # act on all states
      if print_timings:
        print("Act")
      act_start = time.time()
      for states, hit_directions in zip(states_batches, hit_direction_batches):
        for state, hit_direction in zip(states, hit_directions):
          if not is_done(state):
            act(state, hit_direction)

      # run all states until they are waiting for action
      if print_timings:
        print("Run")
      run_start = time.time()
      for i, state_batch in enumerate(states_batches):
        states_batches[i] = pool.map(run_state, state_batch)

      # check if all states in state_batches are done
      if print_timings:
        print("Check")
      check_start = time.time()
      running = any(any(not is_done(state) for state in state_batch)
                    for state_batch in states_batches)

      end_time = time.time()
      if print_timings:
        print(f"Rendering time: {policy_start - render_start:.4f} seconds")
        print(f"Policy time: {act_start - policy_start:.4f} seconds")
        print(f"Act time: {run_start - act_start:.4f} seconds")
        print(f"Run time: {check_start - run_start:.4f} seconds")
        print(f"Check time: {end_time - check_start:.4f} seconds")
        print(f"Total time: {end_time - render_start:.4f} seconds")

    # get the average loss for each batch/model
    losses = [(sum([state_loss(state) for state in state_batch])) /
              len(state_batch) for state_batch in states_batches]
    losses = torch.tensor(losses)

    # apply re-ranking to the losses
    # get sorted indices of the losses
    sorted_indices = torch.argsort(losses).float().to(device)
    losses_reranked = (sorted_indices / (batch_size - 1)) * 2 - 1

    # compute gradient
    grad = [torch.zeros_like(p, device=device) for p in model.parameters()]
    for param_index in range(len(grad)):
      for loss_index, loss in enumerate(losses_reranked):
        grad[param_index] += (loss * noises[loss_index]
                              [param_index]) / (batch_size * standard_dev)

    # apply gradient to the model
    optimizer.zero_grad()
    for param, g in zip(model.parameters(), grad):
      param.grad = g
    optimizer.step()

    # print part of the gradient
    # print(grad[1])

    # find the greatest gradient
    # max_grad = max([torch.max(torch.abs(g)) for g in grad])
    # print(f"Max gradient: {max_grad}")

    return losses, surface_batches

early_exit = False
def signal_handler(sig, frame):
  global early_exit
  early_exit = True
  print("Early exit requested")

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
  use_wandb = True
  states_per_batch = 16
  batch_size = 256
  lr = 1e-3
  standard_dev = 0.01
  random_states = True
  model_type = TinyCNN2
  run_name = "TinyCNN2_1"
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  pool = mp.Pool()

  def state_builder():
    s = make_state(max_strokes=1)
    s['walls'] = []
    return s

  states = [state_builder() for _ in range(states_per_batch)]

  # override default transform
  transform = create_transform(states)

  model = model_type().to(device)
  init_surfaces = None
  best_model_loss, init_surfaces = eval_batched(deepcopy(states), model, device, pool, init_surfaces)
  best_model_loss = sum(best_model_loss) / len(best_model_loss)
  # # find a good initial model
  # print("Looking for a good initial model...")
  # for i in range(100):
  #   print(f"Initial model iteration {i}")
  #   new_model = model_type().to(device)
  #   new_model_loss, _ = eval_batched(deepcopy(states), new_model, device, pool, init_surfaces)
  #   new_model_loss = sum(new_model_loss) / len(new_model_loss)
  #   if new_model_loss < best_model_loss:
  #     model = new_model
  #     best_model_loss = new_model_loss
  #     print(f"Found better model: {best_model_loss}")



  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # create the run directory
  os.makedirs(run_name, exist_ok=True)

  # automatically add the run directory to .gitignore
  with open(".gitignore", "a") as f:
    f.write(f"\n/{run_name}")

  eval_surface = None
  train_surfaces = None

  # save the states
  with open(os.path.join(run_name, "states.pkl"), "wb") as f:
    pickle.dump(states, f)

  # save the transform
  with open(os.path.join(run_name, "transform.pkl"), "wb") as f:
    pickle.dump(transform, f)

  config = {
      "states_per_batch": states_per_batch,
      "batch_size": batch_size,
      "random_states": random_states,
      "model_type": type(model).__name__,
      "optimizer": type(optimizer).__name__,
      "lr": lr,
  }

  if use_wandb:
    wandb.login()
    wandb.init(project="minigolf_es", config=config, name=run_name)

  best_avg_loss = None
  for i in range(2000):
    if early_exit:
      break
    print(f"Training iteration {i}")
    if random_states:
      states = [state_builder() for _ in range(states_per_batch)]
    losses, train_surfaces = train_step(states, model, optimizer, batch_size,
                                        pool, device=device, standard_dev=standard_dev, surface_batches=train_surfaces)
    # sort the losses
    losses = losses.cpu().numpy()
    losses = losses[np.argsort(losses)]
    avg_loss = losses.mean()
    print(f"Avg loss:{avg_loss}")

    # evaluate original model
    loss, eval_surface = eval_batched(
        deepcopy(states), model, device, pool, eval_surface)
    print(f"Original model loss: {sum(loss) / len(loss)}")

    if use_wandb:
      log_info = {
          "avg_loss": avg_loss,
          "min_loss": losses.min(),
          "max_loss": losses.max(),
          "center_loss": sum(loss) / len(loss),
      }
      wandb.log(log_info)

    if i % 20 == 0:
      if best_avg_loss is None or avg_loss < best_avg_loss:
        best_avg_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(run_name, "model_best.pt"))
        with open(os.path.join(run_name, "model_best_losses.txt"), "w") as f:
          f.write(f"Best avg loss: {best_avg_loss}\n")
          f.write(f"Iteration: {i}\n")
      torch.save(model.state_dict(), os.path.join(run_name, "model_latest.pt"))


  pool.close()
  pool.join()
  pool.terminate()

  # save the final model
  print("Saving final model")
  torch.save(model.state_dict(), os.path.join(run_name, "model_final.pt"))
