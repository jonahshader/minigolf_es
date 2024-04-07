from model import BasicCNN
from env import make_state, is_done, step, act
from utils import Vec2
import pygame
import numpy as np
import torch

def eval(model, batch_size=64):
  pass