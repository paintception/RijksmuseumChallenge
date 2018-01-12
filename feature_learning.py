import numpy as np 

import time
import random
import copy

class ValueLearning(object):
	def __init__(self):
		self.n = 3
		self.alpha = 0.99
		self.gamma = 0.99
		self.v = {}

	def make_set_features(self):
		return np.random.randint(10, size = 10)

	def create_initial_state(self, total_set):
		return list(map(lambda _: random.choice(total_set), range(self.n)))

	def update_v(self, state, val):
		self.v[str(state)] = val

	def goto_next_state(self, initial_state):

		action = random.randint(0,1)
		idx = random.randint(0, len(initial_state)-1)
		
		if action == 0:
			initial_state[idx] = initial_state[idx]+1
			return initial_state

		elif action == 1:
			initial_state[idx] = initial_state[idx]-1
			return initial_state

	def get_reward(self):
		return np.random.randint(0,20)	

	def compute_td_value(self, currentS, nextS):

		reward = self.get_reward()
		
		S_0 = str(currentS)
		S_1 = str(nextS)

		if S_0 not in self.v and S_1 not in self.v:
			V_0 = 0 + self.alpha*(reward + self.gamma*(0 - 0))
			self.update_v(S_0, V_0)

		elif S_0 in self.v:	
			if S_1 in self.v:

				V_0 = self.v.get(S_0)
				V_1 = self.v.get(S_1)

				V_0 = V_0 + self.alpha*(reward + self.gamma*(V_1 - V_0))
				self.update_v(S_0, V_0)

			elif S_1 not in self.v:

				V_0 = self.v.get(S_0)
				V_0 = V_0 + self.alpha*(reward + self.gamma*(0 - V_0))
				self.update_v(S_0, V_0)
				
	def main(self):
		total_set = self.make_set_features()
		s = self.create_initial_state(total_set)

		while True:
			state_0 = copy.deepcopy(s)

			next_state = self.goto_next_state(s)
			self.compute_td_value(state_0, next_state)
		
			s = self.create_initial_state(total_set)
			

if __name__ == '__main__':
	VL = ValueLearning()
	VL.main()