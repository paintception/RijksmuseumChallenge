import os
import sys
import glob
import keras
import cv2

import numpy as np 
#import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras import __version__
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import *

from collections import Counter

STORING_PATH = "/data/s2847485/PhD/results/VGG16/"

class TransferLearning(object):
	def __init__(self):
		self.width = 224
		self.height = 224
		self.channels = 3
		self.batch_size = 32
		self.epochs = 100
		self.fc = 1024
		self.layers_to_freeze = 20
		self.activation = "relu"
		self.optimizer = SGD(lr = 0.0001, momentum = 0.9)
		self.loss = "categorical_crossentropy"
		self.early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

	def load_data(self):

		tmp_X = np.load("/data/s2847485/PhD/labels/ImageNet_top_10_artists.npy")
		y = np.load("/data/s2847485/PhD/labels/ImageNet_top_10_labels.npy")
		X = list()

		for image in tmp_X:
			X.append(cv2.resize(image, (self.width, self.height)))
		
		X = np.asarray(X)

		return(X, y)

	def one_hot_encoding(self, y):

		y = [str(i) for i in y]

		self.nb_classes = len(Counter(y).keys())

		encoder = LabelEncoder()
		encoder.fit(y)
		encoded_y = encoder.transform(y)

		final_y = np_utils.to_categorical(encoded_y, self.nb_classes)

		return final_y

	def shape_data(self, images):

		shaped_images = np.reshape(images, (images.shape[0], self.width, self.height, self.channels))

		return shaped_images

	def split_dataset(self, images, labels):
		trainData, testData, trainLabels, testLabels = train_test_split(images, labels, test_size=0.1, random_state=42)

		return [trainData, testData, trainLabels, testLabels]

	def setup_transfer_learning_model(self, model, base_model):	
		for layer in base_model.layers:
			layer.trainable = False

		model.compile(optimizer = "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

	def add_layer(self, base_model):

		tmp_output = base_model.output
		tmp_output = GlobalAveragePooling2D()(tmp_output)
		final_layer = Dense(self.fc, activation = self.activation)(tmp_output)
		predictions = Dense(self.nb_classes, activation = "softmax")(tmp_output)
		model = Model(input = base_model.input, output = predictions)

		return model

	def setup_finetuning(self, model):
		for layer in model.layers[:self.layers_to_freeze]:
			layer.trainable = False
		for layer in model.layers[self.layers_to_freeze:]:
			layer.trainable = True

		model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ["accuracy"])

	def fine_tune(self, training_images, testing_images, training_labels, testing_labels):
		base_model = VGG16(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
		model = self.add_layer(base_model)

		self.setup_transfer_learning_model(model, base_model)

		tl_history = model.fit(training_images, training_labels, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1, validation_data=(testing_images, testing_labels), callbacks = [self.early_stopping], class_weight = "auto")
		model.fit(training_images, training_labels)

		np.save(STORING_PATH+"transfer_learning_accuracies.npy", tl_history.history["val_acc"])

		tl_score = model.evaluate(testing_images, testing_labels, verbose=1)
		print('Test accuracy via Transfer-Learning:', tl_score[1])

		self.setup_finetuning(model)
		
		fine_tuned_history = model.fit(training_images, training_labels, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1, validation_data=(testing_images, testing_labels), callbacks = [self.early_stopping], class_weight = "auto")
		model.fit(training_images, training_labels)
	
		np.save(STORING_PATH+"fine_tuned_accuracies.npy", fine_tuned_history.history["val_acc"])

		ft_score = model.evaluate(testing_images, testing_labels, verbose=1)
		print('Test accuracy after Fine-Tuning:', ft_score[1])

	def main(self):
		X = self.load_data()[0]
		X = self.shape_data(X)
		y = self.load_data()[1]

		y = self.one_hot_encoding(y)

		final_dataset = self.split_dataset(X, y)

		training_images = final_dataset[0]
		testing_images = final_dataset[1]
		training_labels = final_dataset[2]
		testing_labels = final_dataset[3]

		self.fine_tune(training_images, testing_images, training_labels, testing_labels)

if __name__ == '__main__':
	TL = TransferLearning()
	TL.main() 
