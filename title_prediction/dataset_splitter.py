import random
import glob
import time
import os 

IMAGES_PATH = "/home/matthia/Documents/Datasets/Rijksmuseum/jpg2/"

def get_images():
	return(glob.glob(IMAGES_PATH+"*.jpg"))

def store_to_file(name_file, image_set):
	with open(name_file, "w") as file:
		for image in image_set:
			image = os.path.basename(image)
			file.write(image + "\n")

def main():
	images = get_images()
	split_size = int(10*len(images)/100)

	validation_images = images[0:split_size]
	testing_images = images[split_size:split_size*2]
	training_images = images[split_size*2:]

	store_to_file("validation_images.txt", validation_images)
	store_to_file("testing_images.txt", testing_images)
	store_to_file("training_images.txt", training_images)

if __name__ == '__main__':
	main()