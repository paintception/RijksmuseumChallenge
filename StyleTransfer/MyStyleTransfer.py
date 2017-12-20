from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import time
import cv2
import numpy as np 

from keras.applications import vgg19
from keras import backend as K
from matplotlib import pyplot as plt 

CONTENT_IMAGE = "/home/matthia/Desktop/matthia.jpg"
STYLE_IMAGE = "/home/matthia/Desktop/monet.jpg"

class StyleTransferer(object):
	def __init__(self):
		self.total_variation_weight = 1.0
		self.style_weight = 1.0
		self.content_weight = 0.025
		self.epochs = 10
		self.channels = 3
		self.loss_value = None
		self.gradients_value = None
		self.iterations = 2
		self.result_prefix = "output"

	def load_content_img(self):
		content_image = cv2.imread(CONTENT_IMAGE)

		self.width = np.size(content_image, 1)
		self.height = np.size(content_image, 0)

		self.img_nrows = 400
		self.img_ncols = int(self.width * self.img_nrows / self.height)

		return content_image 

	def load_style_img(self):
		style_image = cv2.imread(STYLE_IMAGE)

		return style_image

	def generate_x(self):
		pass

	def create_tensor_representation(self, img):
		img = img_to_array(img)
		img = cv2.resize(img, (self.img_ncols, self.img_nrows))
		img = np.expand_dims(img, axis = 0)
		
		img = vgg19.preprocess_input(img)
		
		return img

	def make_generatedImg_placeholder(self):
		return K.placeholder((1, self.img_nrows, self.img_ncols, 3))

	def load_Vgg19(self, input_tensor):
		model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
		print('VGG19 loaded.')

		return model

	def make_3D_tensor(self, p, a, x):
		input_tensor = K.concatenate([p, a, x], axis = 0)

		return input_tensor

	def gram_matrix(self, x):
		assert K.ndim(x) == 3
		
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
		gram = K.dot(features, K.transpose(features))
		
		return gram

	def style_loss(self, style, combination):
		assert K.ndim(style) == 3
		assert K.ndim(combination) == 3
		
		S = self.gram_matrix(style)
		C = self.gram_matrix(combination)
		size = self.img_nrows * self.img_ncols

		return K.sum(K.square(S - C)) / (4. * (self.channels ** 2) * (size ** 2))

	def content_loss(self, base, combination):
		return K.sum(K.square(combination - base))

	def total_variation_loss(self, x):
		assert K.ndim(x) == 4
		
		a = K.square(x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, 1:, :self.img_ncols - 1, :])
		b = K.square(x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, :self.img_nrows - 1, 1:, :])
		
		return K.sum(K.pow(a + b, 1.25))


	def eval_loss_and_grads(self, x):
		
		x = x.reshape((1, self.img_nrows, self.img_ncols, 3))
		outs = self.f_outputs([x])
		loss_value = outs[0]
		
		if len(outs[1:]) == 1:
			grad_values = outs[1].flatten().astype('float64')
		else:
			grad_values = np.array(outs[1:]).flatten().astype('float64')
		
		return loss_value, grad_values

	def loss(self, x):
		assert self.loss_value is None
		
		loss_value, grad_values = self.eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		
		return grad_values

	def deprocess_image(self, x):
		x = x.reshape((self.img_nrows, self.img_ncols, 3))
		x[:, :, 0] += 103.939
		x[:, :, 1] += 116.779
		x[:, :, 2] += 123.68
		x = x[:, :, ::-1]
		x = np.clip(x, 0, 255).astype('uint8')
		
		return x

	def main(self):

		content_image = self.load_content_img()
		style_image = self.load_style_img()

		p = K.variable(self.create_tensor_representation(content_image))
		a = K.variable(self.create_tensor_representation(style_image))
		x = self.make_generatedImg_placeholder()

		final_tensor = self.make_3D_tensor(p, a, x)
		model = self.load_Vgg19(final_tensor)

		symbolic_model = dict([(layer.name, layer.output) for layer in model.layers])

		loss = K.variable(0.)
		layer_features = symbolic_model['block5_conv2']
		base_image_features = layer_features[0, :, :, :]
		combination_features = layer_features[2, :, :, :]
		loss += self.content_weight * self.content_loss(base_image_features, combination_features)

		feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
		
		for layer_name in feature_layers:
			layer_features = symbolic_model[layer_name]
			style_reference_features = layer_features[1, :, :, :]
			combination_features = layer_features[2, :, :, :]
			sl = self.style_loss(style_reference_features, combination_features)
			loss += (self.style_weight / len(feature_layers)) * sl
		
		loss += self.total_variation_weight * self.total_variation_loss(x)

		grads = K.gradients(loss, x)
		outputs = [loss]

		if isinstance(grads, (list, tuple)):
			outputs += grads
		else:
			outputs.append(grads)

		self.f_outputs = K.function([x], outputs)

		x = self.create_tensor_representation(content_image)

		for i in range(self.iterations):
			print('Start of iteration', i)
			start_time = time.time()
			x, min_val, info = fmin_l_bfgs_b(style_transfer.loss, x.flatten(), fprime=style_transfer.grads, maxfun=20)
			print('Current loss value:', min_val)
			
			img = self.deprocess_image(x.copy())
			fname = self.result_prefix + '_at_iteration_%d.png' % i
			imsave(fname, img)
			end_time = time.time()
			print('Image saved as', fname)
			print('Iteration %d completed in %ds' % (i, end_time - start_time))

if __name__ == '__main__':
	style_transfer = StyleTransferer()
	style_transfer.main()
		