import glob
import nltk
import time 
import pickle
import os 

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt 

import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional

from data_generator import DataGenerator

PATH = "/home/matthia/Documents/Datasets/Flickr1Gb/"
ENCODINGS_PATH = "/home/matthia/Desktop/"

class CaptionPredictor(object):
	def __init__(self):
		self.token = PATH + "captions/Flickr8k.token.txt"
		self.images = PATH + "images/"
		self.train_images_file = PATH + "captions/Flickr_8k.trainImages.txt"
		self.val_images_file = PATH + "captions/Flickr_8k.devImages.txt"
		self.test_images_file = PATH + "captions/Flickr_8k.testImages.txt"
		self.train_encodings = ENCODINGS_PATH + "encoded_images_inceptionV3.ps"
		self.test_encodings = ENCODINGS_PATH + "encoded_images_test_inceptionV3.p"
		self.dataset = ENCODINGS_PATH + "flickr8k_training_dataset.txt"

	def load_captions(self):
		return(open(self.token, 'r').read().strip().split('\n'))

	def process_captions(self, captions):

		d = {}

		for i, row in enumerate(captions):
			row = row.split('\t')
			row[0] = row[0][:len(row[0])-2]
			if row[0] in d:
				d[row[0]].append(row[1])
			else:
				d[row[0]] = [row[1]]

		return d

	def split_data(self, l, img):
		temp = list()
		
		for i in img:
			if i[len(self.images):] in l:
				temp.append(i)
		
		return temp

	def load_images(self):
		return(glob.glob(self.images+'*.jpg'))

	def load_data_info(self, file):
		return(set(open(file, 'r').read().strip().split('\n')))

	def preprocess(self, image_path):
		
		img = image.load_img(image_path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)

		def preprocess_input(x):
			x /= 255.
			x -= 0.5
			x *= 2.
			
			return x

		x = preprocess_input(x)

		return x

	def load_inception_neural_net(self):

		model = InceptionV3(weights='imagenet')

		new_input = model.input
		hidden_layer = model.layers[-2].output

		model_new = Model(new_input, hidden_layer)

		return model_new

	def get_inception_encoding(self, directory, images_set, model):
		
		if os.path.isfile(directory):
			return(pickle.load(open(directory, 'rb')))

		else:
			encoded_dict = {}
			
			for img in tqdm(images_set):

				image = self.preprocess(img)
				temp_enc = model.predict(image)
				temp_enc = np.reshape(temp_enc, temp_enc.shape[1])

				encoded_dict[img[len(self.images):]] = temp_enc

			with open(directory, "wb") as encoded_pickle:
				pickle.dump(encoded_dict, encoded_pickle)

			return encoded_dict

	def process_dict(self, dict_to_process, captions):
		
		processed_dict = {}

		for i in dict_to_process:
			if i[len(self.images):] in captions:
				processed_dict[i] = captions[i[len(self.images):]]

		return(processed_dict)

	def get_corpus(self, dict_to_process):
		
		caps = list()
		unique_words = list()

		for key, val in dict_to_process.items():
			for i in val:
				caps.append('<start> ' + i + ' <end>')

		words = [i.split() for i in caps]

		for word in words:
			unique_words.extend(word)

		return(caps, list(set(unique_words)))

	def word_to_idx(self, unique):
		return({val:index for index, val in enumerate(unique)})

	def idx_to_word(self, unique):
		return({index:val for index, val in enumerate(unique)})

	def get_max_length(self, caps):

		max_len = 0

		for c in caps:
			c = c.split()
			if len(c) > max_len:
				max_len = len(c)

		return max_len

	def get_dataset(self, train_d):
		if os.path.exists(self.dataset):
			return(pd.read_csv(self.dataset, delimiter='\t'))

		else:
			f = open(self.dataset, 'w')
			f.write("image_id\tcaptions\n")

			for key, val in train_d.items():
				for i in val:
					f.write(key[len(self.images):] + "\t" + "<start> " + i +" <end>" + "\n")
			
			f.close()
			
			return(pd.read_csv(self.dataset, delimiter='\t'))

	def main(self):
		original_captions = self.load_captions()
		captions = self.process_captions(original_captions)
		img = self.load_images()

		train_images = self.load_data_info(self.train_images_file)
		train_img = self.split_data(train_images, img)      

		val_images = self.load_data_info(self.val_images_file)
		val_img = self.split_data(val_images, img)

		test_images = self.load_data_info(self.test_images_file)
		test_img = self.split_data(test_images, img)

		inception_neural_net = self.load_inception_neural_net()

		encoding_train = self.get_inception_encoding(self.train_encodings, train_img, inception_neural_net)
		encoding_test = self.get_inception_encoding(self.test_encodings, test_img, inception_neural_net)

		train_d = self.process_dict(train_img, captions)
		val_d = self.process_dict(val_img, captions)
		test_d = self.process_dict(test_img, captions)

		corpus = self.get_corpus(train_d)
		caps = corpus[0]
		unique_trainig_words = corpus[1]
		
		vocab_size = len(unique_trainig_words)

		word2idx = self.word_to_idx(unique_trainig_words)
		idx2word = self.idx_to_word(unique_trainig_words)

		max_len = self.get_max_length(caps)
		dataset = self.get_dataset(train_d)

		c = [i for i in dataset['captions']]
		imgs = [i for i in dataset['image_id']]

		samples_per_epoch = 0
		
		for ca in caps:
			samples_per_epoch += len(ca.split())-1

		trainer = DataGenerator(self.dataset, vocab_size, samples_per_epoch, imgs, encoding_train, max_len, word2idx)
		trainer.train()

c = CaptionPredictor()
c.main()
