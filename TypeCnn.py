from PIL import Image
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

import numpy as np 

import time
import glob
import torch
import cv2

import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.datasets
import torch.utils.data as utils
import torch.optim as optim

IMAGES_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/images/"
LABELS_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/Dataset/type_labels/"

def load_images():
	filelist = glob.glob(IMAGES_PATH+"*.jpg")
	
	return(np.array([np.array(Image.open(fname)) for fname in filelist]))

def load_labels():
	return np.load(LABELS_PATH+"type_labels.npy")

def resize_images(X):
	resized_images = list()
	
	for image in X:
		resized_images.append(cv2.resize(image, (128, 128)))

	return np.asarray(resized_images)

def shape_inputs(X):
	ShapedChessPositions = np.reshape(X, (X.shape[0], 3, 128, 128))
	return ShapedChessPositions

def dataset_split(X, y):
	trainData, testData, trainLabels, testLabels = train_test_split(X, y, test_size=0.1, random_state=42)
	return [trainData, testData, trainLabels, testLabels]

def train(train_loader):
	net = Net()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
	
	print "Ann Called"

	net.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = net(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        
	return F.log_softmax(x)		

def main():

	X = load_images()
	X = resize_images(X)
	X = shape_inputs(X)
	y = load_labels()

	splitted = dataset_split(X, y)

	training_images = np.asarray(splitted[0])
	training_labels = np.asarray(splitted[2])
	testing_images = np.asarray(splitted[1])
	testing_labels = np.asarray(splitted[3])


	print(training_images.shape)

	train_X = torch.from_numpy(training_images).float()
	train_y = torch.from_numpy(training_labels)

	my_training_dataset = utils.TensorDataset(train_X, train_y)
	my_training_dataset = utils.DataLoader(my_training_dataset,batch_size = 128, shuffle=True)

	testing_X = torch.from_numpy(testing_images).float()
	testing_y = torch.from_numpy(testing_labels)
	
	my_testing_set = utils.TensorDataset(testing_X, testing_y)
	my_testing_set = utils.DataLoader(my_testing_set, batch_size = 128, shuffle=False)

	train(my_training_dataset)

if __name__ == '__main__':
	main()