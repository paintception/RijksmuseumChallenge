import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

class VariationalAutoEncoder(object):
	def __init__(self, batch_size, z, channels, height, width):
		self.batch_size = batch_size
		self.z = z
		self.height = height
		self.width = width
		self.channels = channels
		
	def load_data(self):
		return(input_data.read_data_sets('MNIST_data'))

	def lrelu(self, x, alpha=0.3):
		return tf.maximum(x, tf.multiply(x, alpha))

	def encoder(self, X_in, keep_prob):
	
		activation = self.lrelu

		with tf.variable_scope("encoder", reuse=None):
			X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
			x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.contrib.layers.flatten(x)
			
			mean = tf.layers.dense(x, units=self.z)
			std = 0.5 * tf.layers.dense(x, units=self.z)            
			epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.z])) 
			
			z  = mean + tf.multiply(epsilon, tf.exp(std))
			
			return z, mean, std

	def decoder(self, sampled_z, keep_prob, inputs_decoder, reshaped_dim):

		with tf.variable_scope("decoder", reuse=None):
			x = tf.layers.dense(sampled_z, units=inputs_decoder, activation = self.lrelu)
			x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation = self.lrelu)
			x = tf.reshape(x, reshaped_dim)
			x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
			
			x = tf.contrib.layers.flatten(x)
			x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
			img = tf.reshape(x, shape=[-1, 28, 28])
			
			return img

	def train(self):
		
		X_in = tf.placeholder(dtype=tf.float32, shape=[None, self.width, self.height], name='X')
		Y    = tf.placeholder(dtype=tf.float32, shape=[None, self.width, self.height], name='Y')
		Y_flat = tf.reshape(Y, shape=[-1, self.width * self.height])
		keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

		reshaped_dim = [-1, 7, 7, self.channels]

		inputs_decoder = 49 * self.channels / 2

		sampled, mn, sd = self.encoder(X_in, keep_prob)
		dec = self.decoder(sampled, keep_prob, inputs_decoder, reshaped_dim)

		unreshaped = tf.reshape(dec, [-1, 28*28])
		img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
		latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
		loss = tf.reduce_mean(img_loss + latent_loss)
		optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		mnist = self.load_data()

		for i in range(30000):
		    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=self.batch_size)[0]]
		    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
		        
		    if not i % 200:
		        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
		        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
		        plt.show()
		        plt.imshow(d[0], cmap='gray')
		        plt.show()
		        print(i, ls, np.mean(i_ls), np.mean(d_ls))

		randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
		imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
		imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

		for img in imgs:
		    plt.figure(figsize=(1,1))
		    plt.axis('off')
		    plt.imshow(img, cmap='gray')

if __name__ == '__main__':
	VAE = VariationalAutoEncoder(batch_size = 64, channels = 1, z = 9, height = 28, width = 28)
	VAE.train()