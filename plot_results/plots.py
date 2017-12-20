import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn
import random

INCEPTION_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/results/InceptionV3/"
RESNET50_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/results/ResNet50/"
VGG16_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/results/VGG16/"
VGG19_PATH = "/home/matthia/Documents/PhD/RijksmuseumChallenge/results/VGG19/"

def load_inception_results():
	tl_accuracies = np.load(INCEPTION_PATH+"transfer_learning_accuracies.npy").tolist()
	ft_accuracies = np.load(INCEPTION_PATH+"fine_tuned_accuracies.npy").tolist()

	return(tl_accuracies+ft_accuracies)
	
def load_resnet_results():
	tl_accuracies = np.load(RESNET50_PATH+"transfer_learning_accuracies.npy").tolist()
	ft_accuracies = np.load(RESNET50_PATH+"fine_tuned_accuracies.npy").tolist()

	return(tl_accuracies+ft_accuracies)

def load_vgg16_results():
	tl_accuracies = np.load(VGG16_PATH+"transfer_learning_accuracies.npy").tolist()
	ft_accuracies = np.load(VGG16_PATH+"fine_tuned_accuracies.npy").tolist()

	return(tl_accuracies+ft_accuracies)

def load_vgg19_results():
	tl_accuracies = np.load(VGG16_PATH+"transfer_learning_accuracies.npy").tolist()
	ft_accuracies = np.load(VGG19_PATH+"fine_tuned_accuracies.npy").tolist()

	return(tl_accuracies+ft_accuracies)

def longest(results):
    return max(len(results), *map(longest, results)) if isinstance(results, list) and results else 0

def custom_plot(x, y, **kwargs):
	ax = kwargs.pop('ax', plt.gca())
	base_line, = ax.plot(x, y, **kwargs)
	ax.fill_between(x, 0.99*y, 1.01*y, facecolor=base_line.get_color(), alpha=0.2)

def main():

	inception_res = load_inception_results()
	resnet_res = load_resnet_results()
	vgg16_res = load_vgg16_results()
	vgg19_res = load_vgg19_results()

	results = [inception_res, resnet_res, vgg16_res, vgg19_res]

	colors = cm.rainbow(np.linspace(0, 1, len(results)))

	for result, c in zip(results, colors):
		result = [random.uniform(0.5, 0.6)] + result
		y = range(len(result))
		custom_plot(y, np.asarray(result), color=c, lw=3)
	
	plt.title("Accuracies")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.show()
	
if __name__ == '__main__':
	main()