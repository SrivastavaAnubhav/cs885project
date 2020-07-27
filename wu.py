from lattice2d_env import Lattice2DEnv
from gym import spaces
import numpy as np

iterations = 5000000
action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]

alpha = 0.01
epsilon = 0.5
gamma = 0.9

def is_valid(action, env):
	return env._get_adjacent_coords(next(reversed(env.state)))[action] not in env.state

class Wu:
	def __init__(self, seq):
		self.env = Lattice2DEnv(seq, trap_penalty=0)
		self.n = len(seq)
		self.state_num = int((4 ** self.n - 1) / 3)
		self.state_transfer = np.arange(start=2, stop=self.state_num + 1).reshape((-1, 4))
		self.q = np.zeros((self.state_num, 4))

	def oneepisode(self, is_train):
		self.env.reset()
		current_state = 1

		while not self.env.done:
			if is_train:
				if np.random.binomial(1, epsilon) == 1:
					current_action = action_space.sample()
				else:
					current_action = np.argmax(self.q[current_state - 1, :])
			else:
				current_action = np.argmax(self.q[current_state - 1, :])

			for i in range(100):
				if not is_valid(current_action, self.env):
					current_action = action_space.sample()
				else:
					break
			else:
				print(self.env.state)
				exit()

			next_state = self.state_transfer[current_state - 1, current_action]
			next_action = np.argmax(self.q[next_state - 1, :])

			obs, reward, done, info = self.env.step(current_action)
			if reward == -2 and not done:
				# We do not expect to collide when using the rigid criterion
				self.env.render()
				exit()

			if is_train:
				self.q[current_state - 1, current_action] += alpha * (reward + gamma * self.q[next_state - 1, next_action] \
													 - self.q[current_state - 1, current_action])
				if reward != 0:
					assert self.q[current_state - 1, current_action] != 0

			current_state = next_state

		return reward


	def train(self):
		print(self.state_num)
		print(self.state_transfer)
		r_test_max = 0
		r_train_max = 0


		for i in range(iterations):
			r = self.oneepisode(True)
			if r > r_train_max:
				r_train_max = r
				print("best r_train_max:", r)
			if i % 10000 == 0:
				r = self.oneepisode(False)
				if r > r_test_max:
					r_test_max = r
					print("best r_test_max:", r)
				print(i, '/', iterations, 'r=', r)

		r = self.oneepisode(False)
		print("Best reward: ", r)
