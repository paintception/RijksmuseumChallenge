
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

def make_original_pool(total_style_reference_features, total_style_combination_features):

    """
    Create a population of tensors of arbitrary length with nabla set to 10
    """

    reference_population = list()
    combination_population = list()

    dims = total_style_reference_features.get_shape()
    d = dims[-1]

    for individual in xrange(1, 11):    #Let's assume we want a population of 10 individuals
        print("Creating Individual: ", individual)

        single_individual_tensor_reference = list()
        single_individual_tensor_combination = list()

        for feature in xrange(1, 11):     #Let's assume 10 optimal features have to be found

            random_ft = random.randint(0, d-1)

            feat_block = total_style_reference_features[:,:, random_ft]     
            comb_block = total_style_combination_features[:,:, random_ft]

            single_individual_tensor_reference.append(feat_block)
            single_individual_tensor_combination.append(comb_block)

        individual_reference = tf.stack(single_individual_tensor_reference, axis=2) 
        individual_combination = tf.stack(single_individual_tensor_combination, axis=2)

        reference_population.append(individual_reference)
        combination_population.append(individual_combination)

    return(reference_population, combination_population)

def filter_and_make_new_generation(dictionary):

    parents_for_breeding = list()
    new_generation_pool = list()

    sorted_population = OrderedDict(sorted(dictionary.items(), key=itemgetter(1)))
    to_remove = np.mean(sorted_population.values())     #50% is kept

    filtered_population = {k: v for k, v in sorted_population.iteritems() if v <= to_remove}
    parents_for_breeding = [tensor for tensor in filtered_population.keys()]

    def compute_children(parents_for_breeding):
        children = list()
        
        for parent_1, parent_2 in zip(parents_for_breeding[0::2], parents_for_breeding[1::2]):
            features_parent_1 = parent_1[:,:,0:5]
            features_parent_2 = parent_2[:,:,-5:]
            
            children.append(tf.concat([features_parent_1, features_parent_2], 2))
            
        return children

    children = compute_children(parents_for_breeding)
    new_reference_features = parents_for_breeding+children

    new_generation_pool.append(new_reference_features)
    new_combination_features = copy.copy(new_reference_features)
    new_generation_pool.append(new_combination_features)

    return(new_generation_pool)

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

    loss_tracker = list()

    for i in range(10):
        print('Optimizing iteration', i)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        #print('Current loss value:', min_val)
        loss_tracker.append(min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        #imsave("/home/matthia/Desktop/"+fname, img)
        #print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return np.mean(loss_tracker)

x = load_image("mat.jpg")

for layer_name in Learner.feature_layers:

    dictionary = {}

    print("Analysing Layer: ", layer_name)

    layer_features = Learner.outputs_dict['block5_conv2']   #Keep it the same!

    total_style_reference_features = layer_features[1, :, :, :]     #Set corresponding to the total pool of features
                                                                     # our aim is to find the optimal subset in here
    total_style_combination_features = layer_features[2, :, :, :]

    dims = total_style_reference_features.get_shape()
    d = dims[-1]

    original_pool = make_original_pool(total_style_reference_features, total_style_combination_features) 

    print(original_pool)
    print("---------")

    if dims[1] == 25:

        for i in xrange(0, 10):     # number of generations
            
            print("Creating generation: ", i)
          
            total_style_reference_features = original_pool[0]
            total_style_combination_features = original_pool[1]

            del original_pool

            losses = list()

            for style_reference_features, style_combination_features in zip(total_style_reference_features, total_style_combination_features):

                """
                loss = initialize_loss()

                print("Initialized Loss for Individual")

                random_content_features = get_random_content_features(layer_features)

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

                #final_loss = run_experiment(x)
                #losses.append(final_loss)
                """

                loss = random.randint(0,20)
                dictionary[style_reference_features]  = loss

            tmp = filter_and_make_new_generation(dictionary)       
            original_pool = tmp 

#np.save("Loss_behaviour_in_function_of_features.npy", loss_tracker)