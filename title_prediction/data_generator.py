from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional

import pandas as pd 
import numpy as np 

import time

PATH = "/home/matthia/Documents/Datasets/Flickr1Gb/"

class DataGenerator(object):
	def __init__(self, df, vocab_size, samples_per_epoch, imgs, encoding_train, max_len, word2idx):
		self.max_len = max_len
		self.vocab_size = vocab_size
		self.samples_per_epoch = samples_per_epoch
		self.imgs = imgs
		self.encoding_train = encoding_train
		self.word2idx = word2idx
		self.images = PATH + "/images/"
		self.embedding_size = 300
		self.partial_caps = []
		self.next_words = []
		self.images = []	
		self.c = []

	def data_generator(self, batch_size=32):
		partial_caps = []
		next_words = []
		images = []

		df = pd.read_csv('/home/matthia/Desktop/flickr8k_training_dataset.txt', delimiter='\t')
		df = df.sample(frac=1)
		iter = df.iterrows()
		c = []
		imgs = []
		for i in range(df.shape[0]):
			x = next(iter)
			c.append(x[1][1])
			imgs.append(x[1][0])

		count = 0

		while True:
			for j, text in enumerate(c):
				current_image = self.encoding_train[imgs[j]]
				for i in range(len(text.split())-1):
					count+=1
					
					partial = [self.word2idx[txt] for txt in text.split()[:i+1]]
					partial_caps.append(partial)
					
					# Initializing with zeros to create a one-hot encoding matrix
					# This is what we have to predict
					# Hence initializing it with vocab_size length
					n = np.zeros(self.vocab_size)
					# Setting the next word to 1 in the one-hot encoded matrix
					n[self.word2idx[text.split()[i+1]]] = 1
					next_words.append(n)
					
					images.append(current_image)

					if count>=batch_size:
						next_words = np.asarray(next_words)
						images = np.asarray(images)
						partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_len, padding='post')
						yield [[images, partial_caps], next_words]
						partial_caps = []
						next_words = []
						images = []
						count = 0
	
	def train(self):

		image_model = Sequential([Dense(self.embedding_size, input_shape=(2048,), activation='relu'), RepeatVector(self.max_len)])

		caption_model = Sequential([Embedding(self.vocab_size, self.embedding_size, input_length=self.max_len), 
									LSTM(256, return_sequences=True),
									TimeDistributed(Dense(300))])

		final_model = Sequential([Merge([image_model, caption_model], mode='concat', concat_axis=1),
								Bidirectional(LSTM(256, return_sequences=False)),
								Dense(self.vocab_size), Activation('softmax')])

		final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
		final_model.fit_generator(self.data_generator(128), samples_per_epoch=self.samples_per_epoch, nb_epoch=1, verbose=1)
