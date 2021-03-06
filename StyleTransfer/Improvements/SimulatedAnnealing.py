
from __future__ import print_function
from __main__ import *

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from scipy.misc import imsave

from collections import OrderedDict
from operator import itemgetter

from scipy.optimize import fmin_l_bfgs_b

import time
import argparse
import random
import copy

from keras import backend as K

import Learner

import tensorflow as tf
import numpy as np 

width, height = load_img("mat.jpg").size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1.0
result_prefix = "output"

#ground_truth_loss = 3924612000.0
#ground_truth_loss = 50
survival_chance = 70

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        
        return grad_values

def compute_starting_solution(total_style_reference_features, total_style_combination_features):

    """
    We create one single solution which corresponds to a 25x25x10 tensor
    We keep track of the location of the 3rd dimension in order to regulate the temperature later on in the algorithm
    """
    
    single_individual_tensor_reference = list()
    single_individual_tensor_combination = list()

    feature_locations = list()

    dims = total_style_reference_features.get_shape()
    d = dims[-1]

    for feature in xrange(1, 10):     #Let's assume 10 optimal features have to be found

        random_ft = random.randint(0, d-1)
        feature_locations.append(random_ft)

        feat_block = total_style_reference_features[:,:, random_ft]     
        comb_block = total_style_combination_features[:,:, random_ft]

        single_individual_tensor_reference.append(feat_block)
        single_individual_tensor_combination.append(comb_block)

    individual_reference = tf.stack(single_individual_tensor_reference, axis=2) 
    individual_combination = tf.stack(single_individual_tensor_combination, axis=2)

    return(individual_reference, individual_combination, feature_locations)

def load_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img

def initialize_loss():
    return K.variable(0.)

def get_random_content_features(layer_features):
    
    total_content_reference_features = layer_features[0, :, :, :]
    total_combination_content_features = layer_features[2, :, :, :]

    return(total_content_reference_features, total_combination_content_features)

def compute_nabla_loss(total_style_combination_features, loss, i):
    style_reference_features = total_style_combination_features[:,:, 0:i]
    style_combination_features = total_style_combination_features[:,:, 0:i]
        
    sl = prepare_style_loss(style_reference_features, style_combination_features)   # We only minimize on a subset of features with nabla corresponding to i
    loss += (style_weight / len(Learner.feature_layers)) * sl            

    return loss

def gram_matrix(x):
    assert K.ndim(x) == 3
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    
    return gram

def prepare_style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols

    sl = original_style_loss(S, C, channels, size)
    #sl = abs_style_loss(S, C, channels, size)
    #sl = eigenvalues_logs_style_loss(S, C, channels, size)
    #sl = frobenius_distance_style_loss(S, C, channels, size)

    return sl

def original_style_loss(S, C, channels, size):
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def abs_style_loss(S, C, channels, size):
    return K.sum(K.abs(S - C)) / (4. * (channels ** 2) * (size ** 2)) 

def frobenius_distance_style_loss(S, C, channels, size):
    return K.sqrt(K.sum((S - C) * (S - K.transpose(C)))) / (4. * (channels ** 2) * (size ** 2)) 
 
def eigenvalues_logs_style_loss(S, C, channels, size):
    return K.sqrt(K.sum(K.log(S - C))) / (4. * (channels ** 2) * (size ** 2)) 

def original_content_loss(base, combination):
    return K.sum(K.square(combination - base))

def l1_content_loss(base, combination):
    return K.sum(K.abs(combination - base))

def cosine_similarity_content_loss(base, combination):
    num = K.sum(combination - base)
    den = K.sqrt(K.square(K.sum(combination))* K.square(K.sum(base)))

    return num/den      

def radial_basis_content_loss(base, combination):
    sigma = 0.25
    gamma = 1/2*(sigma**2)
       
    return K.exp(-(K.sum(K.abs(combination - base)**2)))

def inverse_variance_combination_content_loss(base, combination):
    num = K.sum(combination/K.var(combination))
    den = K.sum(1/K.var(combination))

    inverse_variance = num/den

    loss = K.sqrt(inverse_variance * K.sum(combination - base))

    return loss

def total_variation_loss(x):
    assert K.ndim(x) == 4
 
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
 
    return K.sum(K.pow(a + b, 1.25))

def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    
    return loss_value, grad_values

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def run_experiment(x):

    for i in range(1):
        print('Optimizing iteration', i)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave("/home/matthia/Desktop/"+fname, img)
        #print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return(min_val)


def create_neighbour_solution(total_style_reference_features, total_style_combination_features, feature_positions):
    pass

x = load_image("mat.jpg")

for layer_name in Learner.feature_layers:

    solution_set = list()

    print("Analysing Layer: ", layer_name)

    content_layer_features = Learner.outputs_dict['block5_conv2']   #Keep it the same!
    style_layer_features = Learner.outputs_dict[layer_name]

    total_style_reference_features = style_layer_features[1, :, :, :]     # Set corresponding to the total pool of features
                                                                                   # our aim is to find the optimal subset in here
    total_style_combination_features = style_layer_features[2, :, :, :]

    original_pool = compute_starting_solution(total_style_reference_features, total_style_combination_features) 

    print("Starting Solution Set is computed")

    S_0_style_reference_features = original_pool[0]
    S_0_style_combination_features = original_pool[1]
    S_0_feature_positions = original_pool[2]

    solution_set.append([S_0_style_combination_features, S_0_style_combination_features])

    while True:

        best_loss = 0
        best_set = None

        create_neighbour_solution(total_style_reference_features, total_style_combination_features, S_0_feature_positions)

        S_1_style_reference_features = original_pool[0]
        S_1_style_combination_features = original_pool[1]
        S_1_feature_positions = original_pool[2]

        solution_set.append([S_1_style_combination_features, S_1_style_reference_features])

        for i in range(0, len(solution_set)-1):

            print("Analysing Set: ", solution_set[i])

            """
            loss = initialize_loss()
            random_content_features = get_random_content_features(content_layer_features)

            sampled_base_image_features = random_content_features[0]
            sampled_combination_image_features =random_content_features[1]

            loss += content_weight * radial_basis_content_loss(sampled_base_image_features, sampled_combination_image_features)

            print("Content Loss computed")

            sl = prepare_style_loss(style_reference_features, style_combination_features)
            loss += (style_weight / len(Learner.feature_layers)) * sl                    

            print("Style Loss computed")

            loss += total_variation_weight * total_variation_loss(Learner.combination_image)
            grads = K.gradients(loss, Learner.combination_image)

            outputs = [loss]

            if isinstance(grads, (list, tuple)):
                outputs += grads
            else:
                outputs.append(grads)

            f_outputs = K.function([Learner.combination_image], outputs)

            evaluator = Evaluator()
            
            final_loss = run_experiment(x)
            """
            
            loss = random.randint(0,20)

            if (best_loss - loss) < 0:
                solution_set.pop(0)
                best_loss = loss
                best_set = solution_set[i]

            else:
                print("Niente")


