import numpy as np
import scipy.io

import tensorflow as tf

VGG = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
WIDTH = 800
HEIGHT = 600
CONTENT_LAYERS = ['relu4_2'] #TODO: compare this too!
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
STYLE_WEIGHTS = [5.0, 4.0, 3.0, 4.0, 5.0] # TODO try this out

def weight(layer):
    vgg_layers = VGG['layers']
    W = vgg_layers[0][layer][0][0][2][0][0]
    return tf.constant(W)

def bias(layer):
    vgg_layers = VGG['layers']
    b = vgg_layers[0][layer][0][0][2][0][1]
    return tf.constant(b.reshape(-1))

def conv(input, layer):
    W = weight(layer)
    b = bias(layer)
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv, b)

def relu(input):
    return tf.nn.relu(input)

def avgpool(input):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def create_model():

    model = {}

    layers = [
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'avgpool1',

        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'avgpool2',

        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'conv3_4', 'relu3_4',
        'avgpool3',

        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'conv4_4', 'relu4_4',
        'avgpool4',

        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'conv5_4', 'relu5_4',
        'avgpool5'
    ]

    # Initial input is the image
    input = tf.Variable(np.zeros((1, HEIGHT, WIDTH, 3)), dtype = 'float32')
    model['input'] = input

    for i,layer in enumerate(layers):

        if 'conv' in layer:
            input = conv(input, i)

        elif 'relu' in layer:
            input = relu(input)

        elif 'avgpool' in layer:
            input = avgpool(input)

        model[layer] = input

    return model

def gram(input, n, m):
    # Reshape to 2D matrix
    matrix = tf.reshape(input, (m, n))
    return tf.matmul(tf.transpose(matrix), matrix)

def style_loss(sess, model):

    loss = 0

    for layer in STYLE_LAYERS:
        for w in STYLE_WEIGHTS:
            # F is generated image
            # P is original image
            F = sess.run(model[layer])
            P = model[layer]

            # Number of filters
            N = F.shape[3]
            # Height x Width of feature map
            M = F.shape[1] * F.shape[2]

            A = gram(P, N, M)
            G = gram(F, N, M)

            E = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2)) * w
            loss += E

    return loss

def content_loss(sess, model):

    loss = 0

    for layer in CONTENT_LAYERS:
        F = sess.run(model[layer])
        P = model[layer]
        N = F.shape[3]
        M = F.shape[1] * F.shape[2]
        loss += (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(F - P, 2))

    return loss
