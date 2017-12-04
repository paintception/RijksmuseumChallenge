from bs4 import BeautifulSoup
from sklearn import preprocessing

import os 
import glob
import time
import re

import numpy as np

METADATA_PATH = "/home/matthia/Documents/PhD/data/metadata/xml2/"
STORING_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/type_labels/"

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
		
	def get_artist_name(self, metadata):
		return(re.findall(self.artist_start+'(.*?)'+self.artist_end, metadata))

	def get_type(self, metadata):
		return(re.findall(self.type_start+'(.*?)'+self.type_end, metadata))		

	def get_material(self, metadata):
		return(re.findall(self.material_start+'(.*?)'+self.material_end, metadata))

	def get_date(self, metadata):
		return(re.findall(self.date_start+'(.*?)'+self.date_end, metadata))

	def encoder(self, labels_to_encode):
		le = preprocessing.LabelEncoder()
		le.fit(labels_to_encode)
		
		return(le.transform(labels_to_encode))

	def create_type_labels(self, labels):
		encoded_type_labels = self.encoder(labels)
		np.save(STORING_PATH+'type_labels.npy', encoded_type_labels)

	def read_xml(self):

		type_labels = list()
		artist_labels = list()

		i = 0

		for subdir, dirs, files in os.walk(METADATA_PATH):
			for file in files:
				if file.endswith(".xml") and i < 100:
					metadata = self.get_metadata(file)
					artist_name = self.get_artist_name(metadata)
					material_name = self.get_material(metadata)
					date_interval = self.get_date(metadata)
					print(date_interval)

					type_labels.append(self.get_type(metadata))
					i += 1

		self.create_type_labels(type_labels)
					
	def main(self):
		self.read_xml()
		
if __name__ == '__main__':
	xml_parser = XML_Parser()
	xml_parser.main()