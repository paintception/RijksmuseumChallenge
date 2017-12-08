from bs4 import BeautifulSoup
from sklearn import preprocessing
from multiprocessing import Pool

import os 
import glob
import time
import re
import cv2

import numpy as np

IMAGES_PATH = "/data/s2847485/PhD/jpg2/"
METADATA_PATH = "/data/s2847485/PhD/xml2/"
STORING_PATH = "/data/s2847485/PhD/labels/"

class XML_Parser(object):
	def __init__(self):
		self.artist_start = '<dc:creator>' 		
		self.artist_end = '</dc:creator>'
		self.type_start = '<dc:type>'
		self.type_end = '</dc:type>'
		self.material_start = '<dc:format>materiaal: '
		self.material_end = '</dc:format>'
		self.date_start = '<dc:date>'
		self.date_end = '</dc:date>'

	def get_metadata(self, f):
		
		handler = open(METADATA_PATH+f).read()
		soup = BeautifulSoup(handler, 'lxml')
		metadata = str(soup.find_all("metadata"))
		
		return(metadata)
	
	def get_images(self):
		images = glob.glob(IMAGES_PATH+"*.jpg")
		images = sorted(images)

		return images
		
	def get_artist_name(self, metadata):
		return(re.findall(self.artist_start+'(.*?)'+self.artist_end, metadata))

	def get_type(self, metadata):
		return(re.findall(self.type_start+'(.*?)'+self.type_end, metadata))		

	def get_material(self, metadata):
		return(re.findall(self.material_start+'(.*?)'+self.material_end, metadata))

	def get_date(self, metadata):
		return(re.findall(self.date_start+'(.*?)'+self.date_end, metadata))

	def check_existing(self, label, filename, mode):
		if label != []:
			return True
		else:
			print("Exception in file: ", filename)
			print("While Looking for: ", mode)

	def encoder(self, labels_to_encode):
		le = preprocessing.LabelEncoder()
		le.fit(labels_to_encode)
		
		return(le.transform(labels_to_encode))

	def image_processer(self, image):
		img = cv2.imread(image)
		img = cv2.resize(img, (28,28))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		return img

	def store_artist_images(self, images, name):
		np.save(STORING_PATH + name, images)
	
	def store_artist_labels(self, labels, name):
		encoded_type_labels = self.encoder(labels)
		np.save(STORING_PATH + name, encoded_type_labels)

	def read_xml(self):

		type_images = list()
		artist_images = list()
		material_images = list()
		date_imahes = list()

		type_labels = list()
		artist_labels = list()
		material_labels = list()
		dates = list()

		images = self.get_images()

		for subdir, dirs, files in os.walk(METADATA_PATH):
			files.sort()
			for (image, file) in zip(images, files):
				if file.endswith(".xml"):
					metadata = self.get_metadata(file)
					artist_name = self.get_artist_name(metadata)
					#type_name = self.get_type(metadata)
					#material_name = self.get_material(metadata)
					#date_interval = self.get_date(metadata)

					if self.check_existing(artist_name, file, "artist_checker"):
						artist_images.append(self.image_processer(image))
						artist_labels.append(artist_name)

					#if self.check_existing(type_name, file, "label_checker"):
						#type_labels.append(type_name)

					#if self.check_existing(material_name, file, "material_checker"):
						#material_labels.append(material_name)

					#if self.check_existing(date_interval, file, "dates_checker"):
						#dates.append(date_interval)
		
		print(len(artist_images))
		print(len(artist_labels))

		self.store_artist_images(artist_images, 'total_artist_images.npy')
		self.store_artist_labels(artist_labels, 'total_artist_labels.npy')

	def main(self):
		self.read_xml()
		
if __name__ == '__main__':
	xml_parser = XML_Parser()
	xml_parser.main()
