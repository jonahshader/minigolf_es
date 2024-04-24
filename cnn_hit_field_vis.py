from model import BasicCNN, ConstModel, BasicCNNNoMag, TinyCNN, TinyCNN2, TinyCNN3
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch
import pickle
import os

from compute_transform import create_transform

from env_render import render_state_tensor, render_state_tensor_for_policy, render_state

def create_hit_field(model, transform, state, autoenc_model, use_policy_render, pixel_size=1, device='cpu'):
  size = state['size']

  hit_field = torch.zeros(2, size // pixel_size, size // pixel_size).to(device)
  
  
  print('Rendering surfaces...')
  for y in range(size // pixel_size):
    surface_tensors = []
    for x in range(size // pixel_size):
      state['ball'].pos = Vec2(x * pixel_size, y * pixel_size)
      if use_policy_render:
        surface_tensor = render_state_tensor_for_policy([state], transform)
      else:
        surface_tensor = render_state_tensor([state], transform)
      surface_tensor = surface_tensor.to(device)
      surface_tensors.append(surface_tensor)
    surface_tensor = torch.cat(surface_tensors, dim=0)

    print('Running model on row...')
    if autoenc_model is not None:
      surface_tensor = autoenc_model(surface_tensor)
    hit_direction = model(surface_tensor)
    # hit_field[:, y, x] = hit_direction[0][:]
    # hit_field = hit_direction.permute(1, 0).reshape(2, size // pixel_size, size // pixel_size)
    hit_field[:, y, :] = hit_direction.permute(1, 0).reshape(2, size // pixel_size)

  return hit_field.cpu()

def render_hit_field(surface, hit_field, pixel_size=1):
  size = hit_field.shape[1]
  for y in range(size):
    for x in range(size):
      hit_direction = hit_field[:, y, x]
      # pygame.draw.line(surface, (255, 0, 0), (x, y), (x + hit_direction[0].item() * 10, y + hit_direction[1].item() * 10))
      # colorize the hit direction, using color wheel
      angle = np.arctan2(hit_direction[1].item(), hit_direction[0].item())
      color = (np.cos(angle) * 127 + 128, np.cos(angle + 2 * np.pi / 3) * 127 + 128, np.cos(angle + 4 * np.pi / 3) * 127 + 128)
      # color in pixel
      pygame.draw.rect(surface, color, pygame.Rect(x * pixel_size, y * pixel_size, pixel_size, pixel_size))


def run(model, transform, make_state_func=make_state, autoenc_model=None, use_policy_render=False, device='cpu'):
  pixel_size = 2
  print('Testing cnn policy...')
  state = make_state_func()
  screen = pygame.display.set_mode(
      (state['size'], state['size']), pygame.SCALED | pygame.RESIZABLE)
  
  print('Creating hit field...')
  hit_field = create_hit_field(model, transform, state, autoenc_model, use_policy_render, pixel_size=pixel_size, device=device)
  print('Rendering hit field...')
  render_hit_field(screen, hit_field, pixel_size=pixel_size)
  render_state(state, screen, no_clear=True)

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
          state = make_state_func()
          print('Creating hit field...')
          hit_field = create_hit_field(model, transform, state, autoenc_model, use_policy_render, pixel_size=pixel_size, device=device)
          print('Rendering hit field...')
          render_hit_field(screen, hit_field, pixel_size=pixel_size)
          render_state(state, screen, no_clear=True)


    pygame.display.flip()
    clock.tick(60)

  pygame.quit()


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  run_name = 'ResModel1_encoded_1'

  # load config
  with open(os.path.join(run_name, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)

  model_type = config['model_type']
  model_args = config['model_args']

  model1 = model_type(**model_args)
  model1.load_state_dict(torch.load(os.path.join(run_name, f'model_final.pt')))
  model1.eval()
  model1.to(device)

  use_policy_render = config['use_policy_render']
  transform = None
  if config['use_autoencoder']:
    # load autoencoder config
    autoencoder_name = config['autoencoder_name']
    with open(os.path.join(autoencoder_name, 'config.pkl'), 'rb') as f:
      autoenc_config = pickle.load(f)

    autoenc_model_type = autoenc_config['model_type']
    autoenc_constructor_args = autoenc_config['constructor_args']
    autoenc_model = autoenc_model_type(**autoenc_constructor_args)
    autoenc_model.load_state_dict(torch.load(os.path.join(autoencoder_name, "model_final.pt")))
    autoenc_model.eval()
    autoenc_model = autoenc_model.encoder
    autoenc_model.to(device)

    # print the number of parameters
    num_params = sum(p.numel() for p in autoenc_model.parameters())
    print(f"Number of encoder parameters: {num_params}")

    use_policy_render = autoenc_config['use_policy_render']
    with open(os.path.join(autoencoder_name, "transform.pkl"), "rb") as f:
      transform = pickle.load(f)
  else:
    autoenc_model = None

  # try loading the transform
  if transform is None:
    try:
      with open(os.path.join(run_name, 'transform.pkl'), 'rb') as f:
        print('Loading transform...')
        transform = pickle.load(f)
    except:
      print('Transform not found, creating new one...')
      transform = create_transform()

  def state_builder():
    s = make_state(max_strokes=4, wall_chance=0.3, wall_overlap=0.4)
    return s
  with torch.no_grad():
    run(model1, transform, state_builder, autoenc_model=autoenc_model, use_policy_render=use_policy_render, device=device)

