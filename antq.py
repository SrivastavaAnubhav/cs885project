from lattice2d_env import Lattice2DEnv
from gym import spaces
import numpy as np

iterations = 10000

delta = 1    # AQ exponent
beta = 2     # heuristic exponent
alpha = 0.1  # learning rate
gamma = 0.3  # discount factor (unnecessary??)


def get_valid_actions(env):
	adj = env._get_adjacent_coords(next(reversed(env.state)))
	return [pair[0] for pair in adj.items() if pair[1] not in env.state]


def get_heuristic(env, actions):
	current_pos = next(reversed(env.state))
	neighbours = env._get_adjacent_coords(current_pos)
	results = np.zeros(len(actions))
	for i, action in enumerate(actions):
		new_state = neighbours[action]
		new_neighbours = env._get_adjacent_coords(new_state)
		for nn in new_neighbours.values():
			if nn in env.state and env.state[nn] == 'H' and nn != current_pos:
				results[i] += 1
	return np.exp(results)


class AntQ:
	def __init__(self, seq):
		self.n = len(seq)
		self.k = self.n  # TODO: change this to self.n
		self.envs = [Lattice2DEnv(seq, trap_penalty=0) for agent in range(self.k)]
		self.state_num = 1 + 4 * (self.n - 1)
		self.aq = np.random.rand(self.state_num, 4)

	def oneepisode(self, env, is_train):
		env.reset()
		current_state = 0
		i = 0
		state_actions = []

		while not env.done:
			valid_actions = get_valid_actions(env)
			aq_vals = self.aq[current_state].take(valid_actions)
			heuristic = get_heuristic(env, valid_actions)

			probs = (aq_vals ** delta) * (heuristic ** beta)
			probs = probs / sum(probs)
			
			if is_train:
				action = np.random.choice(valid_actions, p=probs)
			else:
				# action = np.argmax(self.aq[current_state])
				# if action not in valid_actions:
				# 	action = np.random.choice(valid_actions)
				action = valid_actions[np.argmax(probs)]

			state_actions.append([current_state, action])

			_, reward, done, info = env.step(action)
			if not done:
				assert reward == 0

			next_state = (4 * i) + action + 1
			i += 1

			if is_train:
				self.aq[current_state, action] = (1 - alpha) * self.aq[current_state, action] + \
												 alpha * gamma * max(self.aq[next_state])

			current_state = next_state

		return reward, state_actions

	def train(self):
		grand_best_r = 0
		r_test_max = 0
		for i in range(iterations):
			best_r = 0
			sa = []
			for agent in range(self.k):
				r, state_actions = self.oneepisode(self.envs[agent], True)
				# sa.append(state_actions)
				if r > best_r:
					best_r = r
					best_state_actions = state_actions

			# print("Best reward: ", r)
			# print(best_state_actions)
			for state, action in best_state_actions:
				self.aq[state][action] += alpha * r / self.n

			if i % 100 == 0:
				r, _ = self.oneepisode(self.envs[0], False)
				if r > r_test_max:
					r_test_max = r
					print("best r_test_max:", r)
				print(self.aq)
				print(i, '/', iterations, 'r =', r)

		print("Best reward: ", r)
		print(best_state_actions)
