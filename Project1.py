# MAE 598 Design Optimization - Project # 1
# Benjamin Webb
# 10/21/2022

# Import required libraries
import logging
#import math
#import random as rand
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
#from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define global parameters
FRAME_TIME = np.single(1.0)                 # time inteval
GRAVITY_ACCEL = np.single(9.81 / 1000)      # gravitaional acceleration parameter
BOOST_ACCEL = np.single(14.715 / 1000)      # Thrust accelleration parameter, main engine
BOOST2_ACCEL = np.single(1.4715 / 1000)     # Thrust acceleration parameter, side thrusters

# Define Class for system dynamics
class Dynamics(nn.Module):

	# Initialize class
	def __init__(self):
		super(Dynamics, self).__init__()

	@staticmethod
	def forward(state, action):

		# action: thrust or no thrust
		# action[0]: thrust control, main engine
		# action[1]: thrust control, side thrusters
		# action[2]: omega control

		# State: state variables
		# state[0] = x
		# state[1] = xdot
		# state[2] = y
		# state[3] = ydot
		# state[4] = theta

		# Batch size
		n = state.size(dim=0)

		# Apply gravitational acceleration, only on ydot
		temp_state = torch.tensor([0.0, 0.0, 0.0, -GRAVITY_ACCEL * FRAME_TIME, 0.0], dtype=torch.float)
		delta_gravity = torch.zeros((n, 5), dtype=torch.float)
		delta_gravity = torch.add(input=delta_gravity, other=temp_state)

		# Apply thrust, main engine
		temp_state = torch.zeros((n, 5), dtype=torch.float)
		temp_state[:, 1] = -torch.sin(state[:, 4])
		temp_state[:, 3] = torch.cos(state[:, 4])
		delta_thrust1 = BOOST_ACCEL * FRAME_TIME * torch.mul(temp_state, action[:, 0].reshape(-1, 1))

		# Apply thrust, side thrusters
		temp_state = torch.zeros((n, 5), dtype=torch.float)
		temp_state[:, 1] = torch.cos(state[:, 4])
		temp_state[:, 3] = -torch.sin(state[:, 4])
		delta_thrust2 = BOOST2_ACCEL * FRAME_TIME * torch.mul(temp_state, action[:, 1].reshape(-1, 1))

		# Apply change in theta
		delta_theta = FRAME_TIME * torch.mul(torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0]), action[:, 2].reshape(-1, 1))

		# Combine dynamics
		state = state + delta_gravity + delta_thrust1 + delta_thrust2 + delta_theta

		# Update state vector
		step_mat = torch.tensor([[1.0, FRAME_TIME, 0.0, 0.0, 0.0],
		                         [0.0, 1.0, 0.0, 0.0, 0.0],
		                         [0.0, 0.0, 1.0, FRAME_TIME, 0.0],
		                         [0.0, 0.0, 0.0, 1.0, 0.0],
		                         [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float)

		state = torch.matmul(step_mat, state.T)

		return state.T

# Define Controller Class
class Controller(nn.Module):

	# Initialize class
	def __init__(self, dim_input, dim_hidden, dim_output):
		# dim_input: # of system states
		# dim_output: # of actions
		# dim_hidden: TBD

		super(Controller, self).__init__()
		# Added 2 extra layers
		self.network = nn.Sequential(nn.Linear(dim_input, dim_hidden),
		                             nn.Tanh(), nn.Linear(dim_hidden, dim_hidden),
		                             nn.ELU(),  nn.Linear(dim_hidden, dim_hidden),
		                             nn.GELU(), nn.Linear(dim_hidden, dim_output), nn.Sigmoid())

	# define Controller forward method
	def forward(self, state):
		action = self.network(state)
		return action

# Define Simulation Class
class Simulation(nn.Module):

	# Initialize Class
	def __init__(self, controller, dynamics, T):
		super(Simulation, self).__init__()
		self.state = self.intialize_state()
		self.controller = controller
		self.dynamics = dynamics
		self.T = T
		self.action_trajectory = []
		self.state_trajectory = []

	# Define Simulation class forward method
	def forward(self, state):
		self.action_trajectory = []
		self.state_trajectory = []
		for _ in range(self.T):
			action = self.controller.forward(state)
			state = self.dynamics.forward(state, action)
			self.action_trajectory.append(action)
			self.state_trajectory.append(state)

		return self.error(state)

	@staticmethod
	def intialize_state():
		state = torch.rand((100, 5), dtype=torch.float, requires_grad=False)

		return state

	# Define Simulation class error
	@staticmethod
	def error(state):
		# Sum of the 2-nmom squared all divided by number of batches
		return torch.sum(torch.pow(torch.linalg.vector_norm(state, ord=2, dim=0), 2)) / state.size(dim=0)

# Define Optimizer class. Currently, using LBFGS
class Optimize:

	# Initialize class
	def __init__(self, simulation):
		super(Optimize, self).__init__()
		self.simulation = simulation
		self.parameters = simulation.controller.parameters()
		self.optimizer = optim.LBFGS(self.parameters, lr=0.01)   # Current set learning rate

	# Define Optmize class step function
	def step(self):
		# Define Closure function so gradient can be calculated multiple times
		def closure():
			loss = self.simulation(self.simulation.state)
			self.optimizer.zero_grad()
			loss.backward()

			return loss

		# Possible reccursive operation
		self.optimizer.step(closure)

		return closure()

	# Define Optimize class train function
	def train(self, epochs):
		for epoch in range(epochs):
			loss = self.step()
			print('[%d] loss: %.3f' % (epoch + 1, loss))
			#self.visualize()                                # Will update later

	# Define Optimize class visulize function, will be updated later
	def visualize(self):
		data = np.zeros((self.simulation.T, 4), dtype=np.single)
		for i in range(self.simulation.T):
			temp = self.simulation.state_trajectory[i].detach()
			data[i, :] = temp.numpy()

		x1 = data[:, 0]
		y1 = data[:, 1]
		fig = plt.figure(num=1)
		plt.plot(x1, y1)
		plt.show()

		x2 = data[:, 2]
		y2 = data[:, 3]
		fig = plt.figure(num=2)
		plt.plot(x2, y2)
		plt.show()

# Define main program script
if __name__ == '__main__':

	# Begin timer
	start_time = time.time()

	# Initial test to ensure code is working
	T = 20              # number of time steps
	dim_input = 5       # number of state-space variables, currently 5
	dim_hidden = 30     # size of neurnal network
	dim_output = 3      # number of actions, currently 3

	d = Dynamics()                                      # Created Dynamics class object
	c = Controller(dim_input, dim_hidden, dim_output)   # Created Controller class object
	s = Simulation(controller=c, dynamics=d, T=T)       # Created Simulation class object
	o = Optimize(simulation=s)                          # Created Optimizer Class object
	o.train(epochs=40)                                  # Test code

	# End timer
	end_time = time.time()

	# Print program execution time
	total_time = end_time - start_time
	print('Execution Time:', total_time, 'seconds')
