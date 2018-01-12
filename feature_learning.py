import numpy as np 

import tensorflow as tf 

import time
import random
import copy

from matplotlib import pyplot as plt 

class ValueLearning(object):
	def __init__(self):
		self.n = 3
		self.alpha = 0.99
		self.gamma = 0.99
		self.v = {}
		self.cnt = 0

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

		init_op = tf.initialize_all_variables()

		with tf.Session() as sess:
			sess.run(init_op) #execute init_op
			#print the random values that we sample
			reward = self.get_reward()
			
			S_0 = str(sess.run(currentS))
			S_1 = str(sess.run(nextS))

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

	def make_total_set_tf(self):

		a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
		b = tf.constant([[5.0, 21.0], [3.0, 4.0]])
		c = tf.constant([[6.0, 9.0], [3.0, 4.0]])
		d = tf.constant([[7.0, 8.0], [3.0, 4.0]])

		total_set = tf.stack([a,b,c,d], axis=2)
		dims = total_set.get_shape()
		d = dims[-1]

		return(total_set, d)

	def make_state_tf(self):
		return self.total_set[:,:, random.randint(0, self.d-1)]

	def next_state_tf(self, initial_state):
		while True:
			next_state = self.make_state_tf()
			if initial_state != next_state:
				return next_state
			break

	def main(self):
		#total_set = self.make_set_features()
		#s = self.create_initial_state(total_set)

		tmp = self.make_total_set_tf()

		self.total_set = tmp[0]
		self.d = tmp[1]

		s_0 = self.make_state_tf()

		for i in range(0,200):
			state_0 = tf.Variable(s_0)
			next_state = self.next_state_tf(state_0)
	
			self.compute_td_value(state_0, next_state)
			s_0 = self.make_state_tf()

		print(self.v)

if __name__ == '__main__':
	VL = ValueLearning()
	VL.main()