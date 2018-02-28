from bs4 import BeautifulSoup
from sklearn import preprocessing
from multiprocessing import Pool

import os 
import glob
import time
import re
import cv2

import numpy as np

IMAGES_PATH = "/home/matthia/Documents/Datasets/Rijksmuseum/jpg2/"
METADATA_PATH = "/home/matthia/Documents/Datasets/Rijksmuseum/xml2/"
STORING_PATH = "/home/matthia/Documents/Datasets/Rijksmuseum/txt_files/"

class XML_Parser(object):
	def __init__(self):
		self.start_title = "<dc:title>"
		self.end_title = "</dc:title>"
		self.start_description = "<dc:description>"
		self.end_description = "</dc:description>"

	def get_metadata(self, f):
		
		handler = open(METADATA_PATH+f).read()
		soup = BeautifulSoup(handler, 'lxml')
		metadata = str(soup.find_all("metadata"))
		
		return(metadata)
	
	def get_images(self):
		images = glob.glob(IMAGES_PATH+"*.jpg")
		images = sorted(images)

		return images
		
	def get_description_text(self, metadata):
		return(re.findall(self.start_description+'(.*?)'+self.end_description, metadata))

	def get_title(self, metadata):
		return(re.findall(self.start_title+'(.*?)'+self.end_title, metadata))

	def check_existing(self, label):
		if label != []:
			return True
		else:
			pass

	def read_xml(self):

		images = self.get_images()

		with open('output.txt', 'w') as f:
			for subdir, dirs, files in os.walk(METADATA_PATH):
				for (image, file) in zip(images, sorted(files)):
					if file.endswith(".xml"):
						
						metadata = self.get_metadata(file)
						#info = self.get_description_text(metadata)
						info = self.get_title(metadata)

						if self.check_existing(info):
							image = os.path.basename(image)
							print("Processing Image: ", image)
							print("Relative Metadata: ", file)
							time.sleep(1)
							f.write(image + "\t" + info[0] + "\n")							

	def main(self):
		self.read_xml()
		
if __name__ == '__main__':
	xml_parser = XML_Parser()
	xml_parser.main()