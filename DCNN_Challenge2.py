from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.callbacks import CSVLogger

from collections import Counter

import numpy as np

import keras

IMAGE_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/data/"
LABELS_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/data/"


class CNN(object):
	def __init__(self):
		self.width = 256
		self.height = 256
		self.channels = 1
		self.epochs = 100
		self.batch_size = 128
		self.opt = SGD(lr=0.01)
		self.activation = "elu"
		self.lossFunction = "categorical_crossentropy"

	def load_images(self):
		return np.load(IMAGE_PATH+"testing_images.npy")

	def load_labels(self):
		return np.load(LABELS_PATH+"testing_labels.npy")

	def OneHotEncoding(self, y):

		y = [str(i) for i in self.labels]

		self.classes = len(Counter(y).keys())

		encoder = LabelEncoder()
		encoder.fit(y)
		encoded_y = encoder.transform(y)

		final_y = np_utils.to_categorical(encoded_y, self.classes)

		return final_y

	def shape_data(self):

		shaped_images = np.reshape(self.images, (self.images.shape[0], self.width, self.height, self.channels))

		return shaped_images

	def split_dataset(self):
		trainData, testData, trainLabels, testLabels = train_test_split(self.images, self.labels, test_size=0.1, random_state=42)

		return [trainData, testData, trainLabels, testLabels]

	def NeuralNet(self):
		
		model = Sequential()
		model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(self.width, self.height, self.channels)))
		model.add(Activation(self.activation))
		model.add(Convolution2D(50, 3, 3, border_mode="same")) 
		model.add(Activation(self.activation))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(250, activation=self.activation))
		model.add(Dense(self.classes))
		model.add(Activation('softmax'))

		model.compile(loss=self.lossFunction, optimizer=self.opt, metrics=['accuracy'])

		model.fit(self.training_images, self.training_labels, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1, validation_data=(self.testing_images, self.testing_labels))
		
		model.fit(self.training_images, self.training_labels)

		score = model.evaluate(self.testing_images, self.testing_labels, verbose=0)
		print('Test accuracy:', self.score[1])

		self.model.save_weights('CNNWeights.h5')

	def main(self):
		self.images = self.load_images()
		self.labels = self.load_labels() 
		self.images = self.shape_data()
		self.labels = self.OneHotEncoding(self.labels)
		self.FinalDataset = self.split_dataset()

		self.training_images = self.FinalDataset[0]
		self.testing_images = self.FinalDataset[1]
		self.training_labels = self.FinalDataset[2]
		self.testing_labels = self.FinalDataset[3]

		print "Start of Experiment!"

		self.NeuralNet()

if __name__ == '__main__':

	ConvNet = CNN()
	ConvNet.main()	
