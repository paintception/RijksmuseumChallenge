import numpy as np
import glob

import cv2

IMAGES_PATH = "/data/s2847485/PhD/jpg2/"
RESCALED_PATH = "/data/s2847485/PhD/images/"

def load_images():
	filelist = glob.glob(IMAGES_PATH+"*.jpg")
	filelist = sorted(filelist)

	return filelist

def main():
	images = load_images()	
	processed_images = list()

	for image in images:
		
		img = cv2.imread(image)
		resized_img = cv2.resize(img, (28,28))
		gray_scaled = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
		processed_images.append(gray_scaled)

	np.save("/data/s2847485/PhD/images/images.npy", processed_images)

if __name__== "__main__":
	main()


