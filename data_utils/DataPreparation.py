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

def load_images():
	filelist = glob.glob(IMAGES_PATH+"*.jpg")
	filelist = sorted(filelist)

	return(np.array([np.array(Image.open(fname)) for fname in filelist]))

def load_labels():
	return np.load(LABELS_PATH+"type_labels.npy")

def get_most_common(y, most_common):
	b = Counter(y)
	m = b.most_common(most_common)

	top_types = list()

	for counter in m:
		top_types.append(counter[0])

	return top_types

def store_image(image, tmp_path, i):
	cv2.imwrite(tmp_path+"sample_"+str(i)+".jpg", image)

def main():
	images = load_images()
	labels = load_labels()
	top_types = get_most_common(labels, 2)

	final_labels = list()

	for i, (image, label) in enumerate(zip(images, labels)):
		for top_type in top_types:
			if label == top_type:
				tmp_path = STORING_PATH + "label_"+str(label)+"/"
				if not os.path.exists(tmp_path):
					os.makedirs(tmp_path)

				store_image(image, tmp_path, i)		 
				final_labels.append(label)

	np.save(STORING_PATH+"most_common_labels.npy", final_labels)

if __name__ == '__main__':
	main()