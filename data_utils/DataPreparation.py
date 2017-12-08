import numpy as np 
import glob
import time
import cv2
import os

from collections import Counter
from PIL import Image

IMAGES_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/images/"
LABELS_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/type_labels/"
STORING_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/GAN/"

TOP_THRESH = 10

def load_images():
	return np.load(IMAGES_PATH+"type_images.npy")	

def load_labels():
	return np.load(LABELS_PATH+"type_labels.npy")

def get_most_common(y, most_common):
	b = Counter(y)
	m = b.most_common(most_common)

	top_types = list()

	for counter in m:
		top_types.append(counter[0])

	return top_types

def main():

	most_occurent_imgs = list()
	most_occurent_labels = list()

	images = load_images()
	labels = load_labels()
	top_labels = get_most_common(labels, TOP_THRESH)

	for(image, label) in zip(images, labels):
		for top_label in top_labels:
			if label == top_label:
				most_occurent_imgs.append(image)
				most_occurent_labels.append(label)

	np.save(IMAGES_PATH+"top_images.npy", most_occurent_imgs)
	np.save(LABELS_PATH+"top_labels.npy", most_occurent_labels)

if __name__ == '__main__':
	main()
