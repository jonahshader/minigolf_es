import os
import time
import signal
from model import BasicCNN, ConstModel, BasicCNNNoMag, TinyCNN, TinyCNN2, TinyCNN3, ResModel1
from env import make_state, is_done, step, act, run, state_loss
from utils import Vec2
import numpy as np
import torch
from copy import deepcopy
# for parallel state running
import multiprocessing as mp
import pickle
import wandb

from env_render import render_state_tensor, render_state_tensor_for_policy
from compute_transform import create_transform

print_timings = False
print_eval_timings = False


# closure for running a state
def run_state(state):
  run(state, 1/60)
  return state


def eval_batched(states, model, device, pool, transform=None, use_policy_render=False, encoder_model=None):
  """Eval a single model on a batch of states. Used to evaluate 'center' model."""
  # eval the batch of states
  running = True
  while running:
    # render all states
    render_start = time.time()
    render_fun = render_state_tensor_for_policy if use_policy_render else render_state_tensor
    surfaces = render_fun(states, transform=transform).to(device)

    # if using an autoencoder, encode the surfaces
    if encoder_model is not None:
      surfaces = encoder_model(surfaces)

    # compute hit directions for all states
    policy_start = time.time()
    hit_directions = model(surfaces)
    hit_directions = [Vec2(*d.cpu().numpy().tolist()) for d in hit_directions]

    # act on all states
    act_start = time.time()
    for state, hit_direction in zip(states, hit_directions):
      if not is_done(state):
        act(state, hit_direction)

    # run all states until they are waiting for action
    run_start = time.time()

    # parallelize the state running
    states = pool.map(run_state, states)

    # check if all states are done
    check_start = time.time()
    running = any(not is_done(state) for state in states)

    end_time = time.time()
    if print_eval_timings:
      print(f"Rendering time: {policy_start - render_start:.4f} seconds")
      print(f"Policy time: {act_start - policy_start:.4f} seconds")
      print(f"Act time: {run_start - act_start:.4f} seconds")
      print(f"Run time: {check_start - run_start:.4f} seconds")
      print(f"Check time: {end_time - check_start:.4f} seconds")
      print(f"Total time: {end_time - render_start:.4f} seconds")

  # get the average loss for all states
  return [state_loss(state) for state in states]


def create_mutated_models(model, model_args, batch_size, device, standard_dev):
  # clone the model for each batch
  models = [type(model)(**model_args).to(device) for _ in range(batch_size)]
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

  return models, noises


def train_step(states, model, optimizer, batch_size, pool: mp.Pool, device='cpu', standard_dev=0.01,
               encoder_model=None, use_policy_render=False, model_args={}, transform=None):
  # clone the states for each batch
  states_batches = [deepcopy(states) for _ in range(batch_size)]

  models, noises = create_mutated_models(
      model, model_args, batch_size, device, standard_dev)

  # rewrite to use tensor rendering
  running = True
  while running:
    if print_timings:
      print("Rendering")
    render_start = time.time()
    render_fun = render_state_tensor_for_policy if use_policy_render else render_state_tensor
    # def render_fun_closure(state_batch):
    #   if use_policy_render:
    #     return render_state_tensor_for_policy(state_batch, transform=transform)
    #   else:
    #     return render_state_tensor(state_batch, transform=transform)

    surface_batches = [render_fun(state_batch)
                       for state_batch in states_batches]
    # run in parallel
    # surface_batches = pool.map(render_fun_closure, states_batches)
    surface_batches = [surface_batch.to(device)
                       for surface_batch in surface_batches]

    # if using an autoencoder, encode the surfaces
    if encoder_model is not None:
      surface_batches = [encoder_model(surface_batch)
                         for surface_batch in surface_batches]

    # compute hit directions for all states
    if print_timings:
      print("Policy")
    policy_start = time.time()
    hit_direction_batches = []
    for m, surface_batch in zip(models, surface_batches):
      # TODO: only compute for states that are not done
      hit_direction_batch = m(surface_batch)
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

    # assert all states are done
    if not running:
      for state_batch in states_batches:
        for state in state_batch:
          assert is_done(state), "State is not done"

    end_time = time.time()
    if print_timings:
      print(f"Rendering time: {policy_start - render_start:.4f} seconds")
      print(f"Policy time: {act_start - policy_start:.4f} seconds")
      print(f"Act time: {run_start - act_start:.4f} seconds")
      print(f"Run time: {check_start - run_start:.4f} seconds")
      print(f"Check time: {end_time - check_start:.4f} seconds")
      print(f"Total time: {end_time - render_start:.4f} seconds")

  # get the average loss for each batch/model
  losses = [(sum([state_loss(state, frame_cost=0.0001) for state in state_batch])) /
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

  return losses



early_exit = False


def signal_handler(sig, frame):
  global early_exit
  early_exit = True
  print("Early exit requested")


signal.signal(signal.SIGINT, signal_handler)


def train(config, state_builder):
  use_wandb = config['use_wandb']
  states_per_batch = config['states_per_batch']
  batch_size = config['batch_size']
  iters = config['iters']
  lr = config['lr']
  standard_dev = config['standard_dev']
  random_states = config['random_states']
  model_type = config['model_type']
  model_args = config['model_args']
  run_name = config['run_name']
  device = config['device']
  resume_from = None  # TODO: implement resume_from
  assert run_name is not None, "Please specify a run name"

  use_autoencoder = config['use_autoencoder']
  autoencoder_name = config['autoencoder_name']

  use_policy_render = config['use_policy_render']
  if use_autoencoder:
    with open(os.path.join(autoencoder_name, 'config.pkl'), 'rb') as f:
      autoenc_config = pickle.load(f)

    autoenc_model_type = autoenc_config['model_type']
    autoenc_constructor_args = autoenc_config['constructor_args']
    autoenc_model = autoenc_model_type(**autoenc_constructor_args)
    autoenc_model.load_state_dict(torch.load(
        os.path.join(autoencoder_name, "model_final.pt")))
    autoenc_model.eval()
    autoenc_model = autoenc_model.encoder.to(device)

    # print the number of parameters
    num_params = sum(p.numel() for p in autoenc_model.parameters())
    print(f"Number of encoder parameters: {num_params}")

    use_policy_render = autoenc_config['use_policy_render']
    with open(os.path.join(autoencoder_name, "transform.pkl"), "rb") as f:
      transform = pickle.load(f)
  else:
    autoenc_model = None

  pool = mp.Pool()

  # create the initial batch of states
  # each mutated model will eval on all these
  states = [state_builder() for _ in range(states_per_batch)]
  eval_states = [state_builder() for _ in range(16)]

  # override default transform
  if not use_autoencoder:
    print("No autoencoder, creating transform")
    transform = create_transform(states)

  model = model_type(**model_args).to(device)
  if resume_from is not None:
    print(f"Resuming from {resume_from}")
    model.load_state_dict(torch.load(
        os.path.join(resume_from, "model_latest.pt")))

  # print the number of parameters
  num_params = sum(p.numel() for p in model.parameters())
  print(f"Number of parameters: {num_params}")

  # create the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # create the run directory
  os.makedirs(run_name, exist_ok=True)

  # automatically add the run directory to .gitignore
  with open(".gitignore", "a") as f:
    f.write(f"\n/{run_name}")

  # save the states
  with open(os.path.join(run_name, "states.pkl"), "wb") as f:
    pickle.dump(states, f)

  # save the transform
  with open(os.path.join(run_name, "transform.pkl"), "wb") as f:
    pickle.dump(transform, f)

  # save the config
  with open(os.path.join(run_name, "config.pkl"), "wb") as f:
    pickle.dump(config, f)

  if use_wandb:
    wandb.login()
    wandb_config = {
        **deepcopy(config), 'device': str(device), 'model_type': model_type.__name__}
    wandb.init(project="minigolf_es", config=wandb_config, name=run_name)

  best_avg_loss = None
  for i in range(iters):
    if early_exit:
      break
    print(f"Training iteration {i}")
    if random_states:
      states = [state_builder() for _ in range(states_per_batch)]
    losses = train_step(states, model, optimizer, batch_size, pool,
                        device=device, standard_dev=standard_dev,
                        encoder_model=autoenc_model,
                        use_policy_render=use_policy_render, transform=transform)
    # sort the losses
    losses = losses.cpu().numpy()
    losses = losses[np.argsort(losses)]
    avg_loss = losses.mean()
    print(f"Avg loss:{avg_loss}")

    # evaluate original model
    loss = eval_batched(
        deepcopy(states), model, device, pool, transform=transform, use_policy_render=use_policy_render, encoder_model=autoenc_model)
    print(f"Original model loss: {sum(loss) / len(loss)}")

    eval_loss = eval_batched(
        deepcopy(eval_states), model, device, pool, transform=transform, use_policy_render=use_policy_render, encoder_model=autoenc_model)
    print(f"Eval model loss: {sum(eval_loss) / len(eval_loss)}")

    if use_wandb:
      log_info = {
          "avg_loss": avg_loss,
          "min_loss": losses.min(),
          "max_loss": losses.max(),
          "center_loss": sum(loss) / len(loss),
          "eval_loss": sum(eval_loss) / len(eval_loss),
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


def default_config():
  return {
      'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
      'use_wandb': True,
      'states_per_batch': 1,
      'batch_size': 1024,
      'iters': 2000,
      'lr': 2e-3,
      'standard_dev': 0.01,
      'random_states': True,
      'model_type': ResModel1,
      'model_args': {},
      'run_name': None,
      'resume_from': None,
      # if use_autoencoder is True, this will be set to the value from the autoencoder config
      'use_policy_render': False,
      'use_autoencoder': False,
      'autoencoder_name': "resnet_1",
  }


if __name__ == '__main__':
  config = default_config()
  config['use_wandb'] = True
  config['model_type'] = ResModel1
  config['iters'] = 600
  config['batch_size'] = 256
  config['lr'] = 1e-3
  config['standard_dev'] = 0.005
  config['states_per_batch'] = 32
  config['use_autoencoder'] = True
  config['autoencoder_name'] = 'resnet_1'
  config['run_name'] = 'ResModel1_encoded_2'

  def state_builder():
    s = make_state(max_strokes=1, wall_chance=0.3, wall_overlap=0.4)
    return s

  with torch.no_grad():
    train(config, state_builder=state_builder)
