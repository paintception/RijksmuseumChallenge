from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from keras import backend as K
from kmodes.kmodes import KModes

import numpy as np 

import random
import time

class Clusterer(object):

	def __init__(self):
		self.feature_layers = ['block4_conv1', 'block5_conv1']
		self.p = "mat.jpg" 
		self.a = "munch.jpg"
		self.width, self.height = load_img(self.p).size
		self.n_rows = 400
		self.n_cols = int(self.width * self.n_rows / self.height)
		self.tot_solutions = 10
		self.n_clusters = 3 

	def general_loader(self, image_path):

		img = load_img(image_path, target_size=(self.n_rows, self.n_cols))
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = vgg19.preprocess_input(img)

		return img

	def load_p_image(self):
		return K.variable(self.general_loader(self.p))

	def load_a_image(self):
		return K.variable(self.general_loader(self.a))

	def load_x_image(self):
		return(K.placeholder((1, self.n_rows, self.n_cols, 3)))

	def load_neural_net(self, input_tensor):
		return vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

	def combine_into_tensor(self, p, a, x):
		return K.concatenate([p, a, x], axis=0)

	def get_layers(self, model):
		return(dict([(layer.name, layer.output) for layer in model.layers]))

	def get_feature_maps(self, cnn_layers):
		return(cnn_layers[self.feature_layers[0]])

	def create_pool(self):

		solution_pool = list()
		make_solution = lambda n: [random.randint(0,1) for b in range(0, n)]

		for i in range(0, self.tot_solutions):
			s = make_solution(512)
			if s not in solution_pool:
				solution_pool.append(s)

		return(solution_pool)

	def K_modes_clusterer(self, solution_pool):

		km = KModes(n_clusters=self.n_clusters, init='Huang', n_init=5, verbose=1)
		clusters = km.fit_predict(solution_pool)

		return(clusters)

	def get_clusters(self):
		p = self.load_p_image()
		a = self.load_a_image()
		x = self.load_x_image()

		input_tensor = self.combine_into_tensor(p, a, x)
		neural_net = self.load_neural_net(input_tensor)

		cnn_layers = self.get_layers(neural_net)

		tot_feature_maps = self.get_feature_maps(cnn_layers)

		a_feature_maps = (tot_feature_maps[1, :, :, :])
		p_feature_maps = (tot_feature_maps[2, :, :, :])

		solution_pool = self.create_pool()
		clusters = self.K_modes_clusterer(solution_pool)

		return(solution_pool, clusters)