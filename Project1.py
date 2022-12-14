# MAE 598 Design Optimization - Project # 1
# Benjamin Webb
# 10/21/2022

# Import required libraries
import logging
import numpy as np
import time
import random as rnd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define global parameters
BATCH = np.uint16(250)                     # state variables batch size
FRAME_TIME = np.single(0.1)                 # time inteval
GRAVITY_ACCEL = np.single(0.0981)             # gravitaional acceleration parameter
BOOST_ACCEL = np.single(0.18)      # Thrust accelleration parameter, main engine
BOOST2_ACCEL = np.single(0.018)     # Thrust acceleration parameter, side thrusters
RHO_0 = np.single(1.224)                    # Sea-level air density STA
A1 = np.single(42.6 * 3.66)                 # Reference area 1 for drag calculation
A2 = np.single(np.pi * 1.83**2)             # Reference area 2 for drag calculations
CD = np.single(0.82)                        # Drag coefficient of long cylinder
M = np.single(25000)                        # Mass of rocket in Kg


# Define Class for system dynamics
class Dynamics(nn.Module):

	# Initialize class
	def __init__(self):
		super(Dynamics, self).__init__()

	@staticmethod
	def forward(state, action):

		# action: thrust or no thrust
		# action[0]: thrust control, main engine
		# action[2]: omega control, side thrusters

		# State: state variables
		# state[0] = x
		# state[1] = xdot
		# state[2] = y
		# state[3] = ydot
		# state[4] = theta

		# Apply gravitational acceleration, only on ydot
		delta_gravity = torch.zeros((BATCH, 5), dtype=torch.float)
		delta_gravity = delta_gravity +\
			torch.tensor([0.0, 0.0, 0.0, -GRAVITY_ACCEL * FRAME_TIME, 0.0], dtype=torch.float)

		# Apply thrust, main engine
		tempx = torch.zeros((BATCH, 5), dtype=torch.float)
		tempy = torch.zeros((BATCH, 5), dtype=torch.float)
		tempx.index_fill_(dim=1, index=torch.tensor([1]), value=1)
		tempy.index_fill_(dim=1, index=torch.tensor([3]), value=1)
		tempx = tempx * -torch.sin(state[:, 4].detach().reshape(-1, 1))
		tempy = tempy * torch.cos(state[:, 4].detach().reshape(-1, 1))
		temp = tempx + tempy
		delta_thrust1 = BOOST_ACCEL * FRAME_TIME * torch.mul(temp, action[:, 0].reshape(-1, 1))

		# Apply thrust, side thrusters
		tempx = torch.zeros((BATCH, 5), dtype=torch.float)
		tempy = torch.zeros((BATCH, 5), dtype=torch.float)
		tempx.index_fill_(dim=1, index=torch.tensor([1]), value=1)
		tempy.index_fill_(dim=1, index=torch.tensor([3]), value=1)
		tempx = tempx * torch.cos(state[:, 4].detach().reshape(-1, 1))
		tempy = tempy * -torch.sin(state[:, 4].detach().reshape(-1, 1))
		temp = tempx + tempy
		delta_thrust2 = BOOST2_ACCEL * FRAME_TIME * torch.mul(temp, action[:, 1].reshape(-1, 1))

		# Apply drag
		rho = RHO_0 * torch.exp(-state[:, 2].detach().reshape(-1, 1))
		# Made force negative since they will cause deceleration
		Fdx = -0.5 * CD * A1 * torch.mul(rho, torch.pow(input=state[:, 1].detach().reshape(-1, 1), exponent=2))
		Fdy = -0.5 * CD * A2 * torch.mul(rho, torch.pow(input=state[:, 3].detach().reshape(-1, 1), exponent=2))
		# Acceleration due to drag
		tempx = torch.zeros((BATCH, 5), dtype=torch.float)
		tempy = torch.zeros((BATCH, 5), dtype=torch.float)
		tempx.index_fill_(dim=1, index=torch.tensor([1]), value=1)
		tempy.index_fill_(dim=1, index=torch.tensor([3]), value=1)
		adx = torch.div(torch.mul(Fdx, state[:, 1].detach().reshape(-1, 1)), M)
		ady = torch.div(torch.mul(Fdy, state[:, 3].detach().reshape(-1, 1)), M)

		tempx = torch.mul(tempx, adx)
		tempy = torch.mul(tempy, ady)
		temp_state = tempx + tempy

		delta_drag = torch.mul(temp_state, FRAME_TIME)

		# Apply change in theta
		delta_theta = FRAME_TIME * torch.mul(torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0]), action[:, 1].reshape(-1, 1))

		# Combine dynamics
		state = state + delta_gravity + delta_thrust1 + delta_thrust2 + delta_drag + delta_theta

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
		                             nn.ELU(alpha=0.5), nn.Linear(dim_hidden, dim_output), nn.Softsign())

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
		# Batch
		rand = np.zeros((BATCH, 5), dtype=np.single)
		for i in range(BATCH):
			rand[i, 0] = rnd.uniform(0.0, 0.2)
			rand[i, 1] = rnd.uniform(0.0, 0.2)
			rand[i, 2] = rnd.uniform(0.8, 1.0)
			rand[i, 3] = rnd.uniform(0.0, 0.2)
			rand[i, 4] = rnd.uniform(-0.1, 0.1)

		state = torch.tensor(data=rand, dtype=torch.float, requires_grad=False)

		return state

	# Define Simulation class error
	@staticmethod
	def error(state):
		# Sum of squares all divided by number of batches
		return torch.sum(torch.pow(input=state, exponent=2)) / np.single(BATCH)

# Define Optimizer class. Currently, using LBFGS
class Optimize:

	# Initialize class
	def __init__(self, simulation):
		super(Optimize, self).__init__()
		self.simulation = simulation
		self.parameters = simulation.controller.parameters()
		self.optimizer = optim.LBFGS(self.parameters, lr=0.01)
		# Implementing dynamic learning rate
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=True)
		# Parameter for plotting
		self.best_loss = torch.tensor(np.inf, dtype=torch.float, requires_grad=False)
		self.best_state = torch.zeros((self.simulation.T, 5), dtype=torch.float, requires_grad=False)
		self.best_action = torch.zeros((self.simulation.T, 2), dtype=torch.float, requires_grad=False)

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
			self.scheduler.step(metrics=loss)
			print('[%d] loss: %.3f' % (epoch + 1, loss))
			if loss < self.best_loss:
				self.best_loss = loss
				temp = self.simulation.state_trajectory[-1].detach()
				(minx, idx) = torch.min(torch.linalg.vector_norm(temp, ord=2, dim=1).reshape(-1, 1), dim=0)
				for i in range(self.simulation.T):
					temp_state = self.simulation.state_trajectory[i].detach()
					temp_action = self.simulation.action_trajectory[i].detach()
					self.best_state[i, :] = temp_state[idx, :]
					self.best_action[i, :] = temp_action[idx, :]

			if epoch == epochs - 1:
				self.visualize()

	# Define Optimize class visulize function
	def visualize(self):
		data = np.array(self.best_state.detach())
		data2 = np.array(self.best_action.detach())
		t = np.arange(0.1, 20.1, 0.1)

		x1 = data[:, 0]
		y1 = data[:, 1]
		plt.figure(num=1)
		plt.plot(x1, y1)
		plt.grid(visible=True, which='both', axis='both')
		plt.title('State With Lowest Loss Function')
		plt.ylabel('xdot')
		plt.xlabel('x')
		plt.show()

		x1 = data[:, 2]
		y1 = data[:, 3]
		plt.figure(num=2)
		plt.plot(x1, y1)
		plt.grid(visible=True, which='both', axis='both')
		plt.title('State With Lowest Loss Function')
		plt.ylabel('ydot')
		plt.xlabel('y')
		plt.show()

		x1 = t
		y1 = data[:, 4]
		plt.figure(num=3)
		plt.plot(x1, y1)
		plt.grid(visible=True, which='both', axis='both')
		plt.title('State With Lowest Loss Function')
		plt.ylabel('phi')
		plt.xlabel('t')
		plt.show()

		x1 = t
		y1 = data2[:, 0]
		plt.figure(num=4)
		plt.plot(x1, y1)
		plt.grid(visible=True, which='both', axis='both')
		plt.title('State With Lowest Loss Function')
		plt.ylabel('Main Thruster Control State')
		plt.xlabel('t')
		plt.show()

		x1 = t
		y1 = data2[:, 1]
		plt.figure(num=5)
		plt.plot(x1, y1)
		plt.grid(visible=True, which='both', axis='both')
		plt.title('State With Lowest Loss Function')
		plt.ylabel('Side Thrusters Control State')
		plt.xlabel('t')
		plt.show()


# Define main program script
if __name__ == '__main__':

	# Begin timer
	start_time = time.time()

	# Initial test to ensure code is working
	T = 200             # number of time steps
	dim_input = 5       # number of state-space variables, currently 5
	dim_hidden = 500    # size of neurnal network
	dim_output = 2      # number of actions, currently 2

	d = Dynamics()                                      # Created Dynamics class object
	c = Controller(dim_input, dim_hidden, dim_output)   # Created Controller class object
	s = Simulation(controller=c, dynamics=d, T=T)       # Created Simulation class object
	o = Optimize(simulation=s)                          # Created Optimizer Class object
	o.train(epochs=40)                                  # Test code

	# End timer
	end_time = time.time()

	# Print Final state trajectory and action trajectory
	print('Best State:')
	print(o.best_state[T-1, :])
	print(' ')
	print('Action Controls:')
	print(o.best_action[T-1, :])
	print(' ')

	# Print program execution time
	total_time = end_time - start_time
	print('Execution Time:', total_time, 'seconds')
