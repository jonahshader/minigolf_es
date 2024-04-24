import pygame
from env import make_state, is_done, step, act, run, state_loss, load_states
from copy import deepcopy
import numpy as np
import multiprocessing as mp

from env_render import render_state
from utils import Vec2

multithread = False


def run_and_eval(state, action, eval_action_count=None):
  if eval_action_count is None:
    eval_action_count = action.shape[0]
  for i in range(min(action.shape[0], eval_action_count)):
    act(state, Vec2(action[i][0], action[i][1]))
    run(state, 1/60)
    if is_done(state):
      break
  return state_loss(state, bounce_cost=0, frame_cost=0.0001)


def ga_simple(state, action: np.ndarray, n: int, loss: float, noise: float, eval_action_count=None, pool: mp.Pool = None) -> tuple[np.ndarray, float]:
  # create n copies of state
  states = [deepcopy(state) for _ in range(n)]

  # action is a matrix of shape (p, 2), where p is the number of strokes for par

  # we want n of these, so the desired shape is (n, p, 2)
  actions = np.tile(action, (n, 1, 1))

  # add noise to the actions
  actions += np.random.normal(0, noise, actions.shape)

  # ensure the norm of each action is less than or equal to 1
  norms = np.linalg.norm(actions, axis=2)
  large_norms = norms > 1
  actions[large_norms] /= norms[large_norms][:, np.newaxis]

  # run each state with the corresponding action
  # TODO: multithread this
  losses = [run_and_eval(states[i], actions[i], eval_action_count)
            for i in range(n)]
  # pick the best loss
  best_loss = min(losses)
  if best_loss < loss:
    best_idx = losses.index(best_loss)
    return actions[best_idx], best_loss
  else:
    return action, loss
  

handmade_states = load_states()

use_handmade_states = True

if use_handmade_states:
  current_state = 0
  def state_builder():
    global current_state
    state = deepcopy(handmade_states[current_state])
    state['ball_start'] = deepcopy(state['ball'].pos)
    current_state = (current_state + 1) % len(handmade_states)
    return state
else:
  def state_builder():
    return make_state(max_strokes=4, num_surfaces=3, max_surface_size=128)

if __name__ == '__main__':
  p = 4
  iterations = 64
  population_size = 16
  initial_population_size = 128

  state = state_builder()
  # render the playthrough
  screen = pygame.display.set_mode(
      (state['size'], state['size']), pygame.SCALED | pygame.RESIZABLE)

  clock = pygame.time.Clock()
  
  while True:
    state = state_builder()
    render_state(state, screen, extras=True)
    pygame.display.flip()
    action = np.random.normal(0, 0.1, (p, 2))

    # compute the loss of the initial action
    loss = run_and_eval(deepcopy(state), action)
    print(f'Initial loss: {loss}')

    # run with higher initial population size
    action, loss = ga_simple(
        state, action, initial_population_size, loss, 2)
    print(f'Initial population loss: {loss}')

    # run the genetic algorithm
    for i in range(iterations):
      action, loss = ga_simple(
          state, action, population_size, loss, 1 / (i * 0.2 + 1))
      print(f'Iteration {i}, loss: {loss}')
      if loss <= 1.1:
        break

    print(f'Final action:\n{action}, loss: {loss}')


    running = True
    action_index = 0
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
          print('Quitting')
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            running = False

      if is_done(state):
        running = False
        print('Done')

      if step(state, 1/60):
        if action_index < p:
          act(state, Vec2(*action[action_index]))
          action_index += 1
        else:
          print('Ran out of actions')
          running = False

      render_state(state, screen, extras=True)
      pygame.display.flip()
      clock.tick(45)
