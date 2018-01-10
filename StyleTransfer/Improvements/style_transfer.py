from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import itertools
import sys

from keras.models import *
from keras.layers import *

from keras.applications import vgg19
from keras import backend as K

width, height = load_img("NewYork.jpg").size
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

def get_base_image():
    return K.variable(load_image("NewYork.jpg"))

def get_style_image():
    return K.variable(load_image("munch.jpg"))

def initialize_loss():
    return K.variable(0.)

def load_model():
    model = vgg19.VGG19(input_tensor=input_tensor, include_top=False)

    tmp_output = model.output
    tmp_output = GlobalAveragePooling2D()(tmp_output)
    final_layer = Dense(1024, activation = "relu")(tmp_output)
    predictions = Dense(20, activation = "softmax")(tmp_output)
    model = Model(input = model.input, output = predictions)
   
    model.load_weights("/data/s2847485/PhD/results/VGG19/VGG19_fine_tuned_weights.h5")

    return model

def combine_into_tensor(base_image, style_reference_image, combination_image):
    return K.concatenate([base_image, style_reference_image, combination_image], axis=0)

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
    return K.exp(-0.90*(K.sum(K.abs(combination - base)**2)))

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

def run_experiment(x, feature_layers):

    loss = list()
    directory = "/data/s2847485/PhD/style_transfer_results/feature_search/"+str(feature_layers)+"/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(100):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        #print('Current loss value:', min_val)
        loss.append(min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(directory+fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        #print('Iteration %d completed in %ds' % (i, end_time - start_time))

    np.save(directory+"neural_net_loss.npy", loss)

base_image = get_base_image()
style_reference_image = get_style_image()

combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = combine_into_tensor(base_image, style_reference_image, combination_image)
model = load_model()
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = initialize_loss()

layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

# Decide which kind of content loss 

loss += content_weight * original_content_loss(base_image_features, combination_features)
#loss += content_weight * l1_content_loss(base_image_features, combination_features)
#loss += content_weight * radial_basis_content_loss(base_image_features, combination_features)
#loss += content_weight * cosine_similarity_content_loss(base_image_features, combination_features)
#loss += content_weight * inverse_variance_combination_content_loss(base_image_features, combination_features)

"""
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
"""

if len(sys.argv) == 3:
    tmp_feature_layers = (sys.argv[1], sys.argv[2])

elif len(sys.argv) == 4:
    tmp_feature_layers = (sys.argv[1], sys.argv[2], sys.argv[3])

elif len(sys.argv) == 5:
    tmp_feature_layers = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

elif len(sys.argv) == 6:
    tmp_feature_layers = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

ft = list(tmp_feature_layers)
feature_layers = [x.strip("'") for x in ft]

for layer_name in feature_layers:
    
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = prepare_style_loss(style_reference_features, combination_features)
            
    loss += (style_weight / len(feature_layers)) * sl
    
loss += total_variation_weight * total_variation_loss(combination_image)
grads = K.gradients(loss, combination_image)

outputs = [loss]

if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

evaluator = Evaluator()
x = load_image("NewYork.jpg")

run_experiment(x, feature_layers)
 
