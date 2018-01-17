
from __future__ import print_function
from __main__ import *

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import random

from keras import backend as K

import Learner

import tensorflow as tf

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

def load_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img

def initialize_loss():
    return K.variable(0.)

def get_random_content_features(layer_features):
    
    original_content_feat = list()
    random_content_feat = list()

    total_content_reference_features = layer_features[0, :, :, :]
    total_combination_content_features = layer_features[2, :, :, :]

    """
    dims = total_content_reference_features.get_shape()
    d = dims[-1]

    for i in range(0, 4):

        random_ft = random.randint(0, d-1)

        feat_block_or = total_content_reference_features[:,:, random_ft]
        feat_block_random = total_combination_content_features[:,:, random_ft]

        original_content_feat.append(feat_block_or)
        random_content_feat.append(feat_block_random)

    content_reference_features = tf.stack(original_content_feat, axis = 2)
    random_content_features =  tf.stack(random_content_feat, axis = 2)
    """
    #return(content_reference_features, random_content_features)

    return(total_content_reference_features, total_combination_content_features)

def prepare_style_features(original_initial_style_block, random_initial_style_block):
    style_reference_features = tf.stack(original_initial_style_block, axis = 2)

    print(style_reference_features)
    time.sleep(2)

    combination_features =  tf.stack(random_initial_style_block, axis = 2)     

    sl = prepare_style_loss(style_reference_features, combination_features)

    return sl

def get_style_loss_and_state(loss):
    
    for layer_name in Learner.feature_layers:

        original_initial_style_block_1 = list()
        random_initial_style_block_1 = list()

        original_initial_style_block_2 = list()
        random_initial_style_block_2 = list()

        layer_features = Learner.outputs_dict[layer_name]
        
        total_style_reference_features = layer_features[1, :, :, :]
        total_style_combination_features = layer_features[2, :, :, :]

        dims = total_style_reference_features.get_shape()
        d = dims[-1]

        for i in range(0, 4):   # Creation of what we consider as S_0

            random_ft = random.randint(0, d-1)

            feat_block = total_style_reference_features[:,:, random_ft]     # These features correspond to S_0
            comb_block = total_style_combination_features[:,:, random_ft]   # These features correspond to S_0
        
            receptive_field = feat_block.get_shape()

            if receptive_field[1] == 25:
                original_initial_style_block_1.append(feat_block)
                random_initial_style_block_1.append(comb_block)

            elif receptive_field[1] == 50:
                original_initial_style_block_2.append(feat_block)
                random_initial_style_block_2.append(comb_block)

        if len(original_initial_style_block_2) == 4:
        
            sl = prepare_style_features(original_initial_style_block_2, random_initial_style_block_2)
            loss += (style_weight / len(Learner.feature_layers)) * sl

        if len(original_initial_style_block_1) == 4:

            sl = prepare_style_features(original_initial_style_block_1, random_initial_style_block_1)
            loss += (style_weight / len(Learner.feature_layers)) * sl

    return loss

def gram_matrix(x):
    assert K.ndim(x) == 3
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    
    return gram

def prepare_style_loss(style, combination):
    # Decide which kind of style loss

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
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        loss_tracker.append(min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave("/home/matthia/Desktop/"+fname, img)
        end_time = time.time()
        #print('Iteration %d completed in %ds' % (i, end_time - start_time))

    #np.save("/data/s2847485/PhD/style_transfer_results/eigenvalue_style_loss/eigenvalue_style_loss.npy", loss)
    return loss_tracker

loss = initialize_loss()

layer_features = Learner.outputs_dict['block5_conv2']   #Keep it the same!

random_content_features = get_random_content_features(layer_features)

sampled_base_image_features = random_content_features[0]
sampled_combination_image_features =random_content_features[1]

# Decide which kind of content loss 

#loss += content_weight * original_content_loss(base_image_features, combination_features)
#loss += content_weight * l1_content_loss(base_image_features, combination_features)
loss += content_weight * radial_basis_content_loss(sampled_base_image_features, sampled_combination_image_features)
#loss += content_weight * cosine_similarity_content_loss(base_image_features, combination_features)
#loss += content_weight * inverse_variance_combination_content_loss(base_image_features, combination_features)

loss = get_style_loss_and_state(loss)

loss += total_variation_weight * total_variation_loss(Learner.combination_image)
grads = K.gradients(loss, Learner.combination_image)

outputs = [loss]

if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([Learner.combination_image], outputs)

evaluator = Evaluator()

x = load_image("mat.jpg")

loss = run_experiment(x)