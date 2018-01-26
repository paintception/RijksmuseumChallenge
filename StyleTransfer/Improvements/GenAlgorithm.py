
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
survival_chance = 80

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

    for individual in xrange(0, 20):    #Let's assume we want a population of 10 individuals

        single_individual_tensor_reference = list()
        single_individual_tensor_combination = list()

        for feature in xrange(0, 10):     #Let's assume 10 optimal features have to be found

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

def filter_and_make_new_generation(f_dict, ref_dict, total_style_reference_features, total_content_reference_features, tmp_style_features, tmp_content_features):

    print(f_dict)
    print(len(f_dict))

    """
    def remove_duplicate_soultions():
            raw_population = {}

            for key,value in dictionary.items():
                if value not in raw_population.values():
                    raw_population[key] = value

            return(raw_population)
    
    raw_population = remove_duplicate_soultions()

    print(raw_population)

    """

    ground_truth_loss = max(f_dict.values())

    parents_for_breeding = list()
    new_generation_pool = list()

    new_reference_features = list()

    survival_threshold = (survival_chance*ground_truth_loss)/100

    filtered_population = {k: v for k, v in f_dict.iteritems() if v >= survival_threshold}
    filtered_references = {k: v for k, v in ref_dict.iteritems() if v >= survival_threshold}

    parents_for_breeding = [tensor for tensor in filtered_population.keys()]
    references_for_breeding = [tensor for tensor in filtered_references.keys()]

    print("Beginning of breeding")

    def compute_children_features(parents_for_breeding):

        children_features = list()

        for parent_1, parent_2 in zip(parents_for_breeding[0::2], parents_for_breeding[1::2]):
            features_parent_1 = parent_1[:,:,0:5]
            features_parent_2 = parent_2[:,:,-5:]
            
            children_features.append(tf.concat([features_parent_1, features_parent_2], 2))

        return children_features

    def compute_reference_features(references_for_breeding):

        reference_features = list()

        for parent_1, parent_2 in zip(references_for_breeding[0::2], references_for_breeding[1::2]):  
            features_parent_1 = parent_1[:,:,0:5]
            features_parent_2 = parent_2[:,:,-5:]
            
            reference_features.append(tf.concat([features_parent_1, features_parent_2], 2))

        return reference_features


    print("End of breeding")

    children_features = compute_children_features(parents_for_breeding)
    reference_features = compute_reference_features(references_for_breeding) 

    tmp_f = list()
    tmp_r = list()

    tmp_f.append(parents_for_breeding)
    tmp_f.append(children_features)
    tmp_r.append(references_for_breeding)
    tmp_r.append(reference_features)

    new_features_population = [item for sublist in tmp_f for item in sublist]
    new_reference_features = [item for sublist in tmp_r for item in sublist]

    """
    #dims = total_style_reference_features.get_shape()
    #d = dims[-1]

    def make_random_mutation(new_reference_features):

        mutated_features = list()

        for solution in new_reference_features:

            random_feat = random.randint(0, d-1)
            random_f_individual = total_style_reference_features[:,:, random_feat]
            
            len_ = solution.get_shape()

            mutated_solution = solution[:,:,0:len_[2]-1]

            tmp = (tf.expand_dims(random_f_individual, 2))
            mutated_features.append(tf.concat([mutated_solution, tmp], axis = 2 ))

        return mutated_features

    mutated_features = make_random_mutation(new_reference_features)

    mutated_features = filtered_population
    final_population.extend(mutated_features)
    #final_population.extend(filtered_population)

    new_reference_features = raw_population

    """

    def make_random_individual():
        random_individual = list()
        random_ref = list()

        dims = total_style_reference_features.get_shape()
        d = dims[-1]

        for i in range(0,10):
            random_feat = random.randint(0, d-1)
            
            random_f_individual = total_style_reference_features[:,:, random_feat]     
            random_r_individual = total_content_reference_features[:,:, random_feat]

            random_individual.append(random_f_individual)
            random_ref.append(random_r_individual)

        return(tf.stack(random_individual, axis = 2), tf.stack(random_ref, axis = 2))

    if len(new_features_population) < len(f_dict):
        while(True):

            print("replacing")

            random_individuals = list()
            random_references = list()

            new = make_random_individual()

            random_individuals.append(new[0])
            random_references.append(new[1])

            new_features_population.extend(random_individuals)
            new_reference_features.extend(random_references)

            if len(f_dict) == len(new_features_population):
                break

    new_generation_pool.append(new_features_population)
    new_generation_pool.append(new_reference_features)

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

    for i in range(1):
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave("/home/matthia/Desktop/"+fname, img)
        #print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return(min_val)

x = load_image("mat.jpg")

for layer_name in Learner.feature_layers:

    print("Analysing Layer: ", layer_name)

    content_layer_features = Learner.outputs_dict['block5_conv2']   #Keep it the same!
    style_layer_features = Learner.outputs_dict[layer_name]

    original_total_style_reference_features = style_layer_features[1, :, :, :]     # Set corresponding to the total pool of features
                                                                                   # our aim is to find the optimal subset in here
    original_total_style_combination_features = style_layer_features[2, :, :, :]

    dims = original_total_style_reference_features.get_shape()
    d = dims[-1]

    original_pool = make_original_pool(original_total_style_reference_features, original_total_style_combination_features) 

    if dims[1] == 50:
        for i in xrange(0, 20):     # number of generations
            
            print("Computing generation: ", i)
            time.sleep(1)
          
            total_style_reference_features = original_pool[0]
            total_style_combination_features = original_pool[1]

            del original_pool
            
            feat_dictionary = {}
            ref_dictionary = {}

            for style_reference_features, style_combination_features in zip(total_style_reference_features, total_style_combination_features):
                
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

                del loss
                del evaluator

                #final_loss = random.randint(0,100)
                
                feat_dictionary[style_reference_features]  = final_loss
                ref_dictionary[style_combination_features] = final_loss    

            tmp = filter_and_make_new_generation(feat_dictionary, ref_dictionary, original_total_style_reference_features, original_total_style_combination_features, total_style_reference_features, total_style_combination_features)    #send the ones picked up in generation 1

            #del dictionary

            original_pool = tmp 

        #np.save("Loss_behaviour_in_function_of_features.npy", loss_tracker)