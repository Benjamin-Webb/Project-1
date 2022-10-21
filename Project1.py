# MAE 598 Design Optimization - Project # 1
# Benjamin Webb
# 10/21/2022

# Import required libraries
import logging
import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define global parameters
FRAME_TIME = 0.1        # time inteval
GRAVITY_ACCEL = 0.12    # gravitaional acceleration parameter
BOOST_ACCEL = 0.18      # Trust accelleration parameter

# Define Class for system dynamics
class Dynamics(nn.Module):

	# Initialize class
	def __init__(self):
		super(Dynamics, self).__init__()

	@staticmethod
	def forward(state, action):

		# action: thrust or no thrust

		# Apply gravitational acceleration
		delta_gravity = torch.tensor([0.0, GRAVITY_ACCEL * FRAME_TIME], dtype=torch.double)

		# Apply thrust
		delta_thrust = BOOST_ACCEL * FRAME_TIME * torch.tensor([0.0, -1.0], dtype=torch.double) * action

		# Update state vector
		state = state + delta_thrust + delta_gravity
