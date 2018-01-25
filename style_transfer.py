import os
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

VGG = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
LEARNING_RATE = 3.0
ITERATIONS = 100
ALPHA = 5
BETA = 100

CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

STYLE = 'img/style/vangogh.jpg'
CONTENT = 'img/content/sunflower.jpg'
WIDTH = 800
HEIGHT = 600

# VGG-19 mean RGB
RGB_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

# TODO: move these out?
def white_noise(content):
    noise = np.random.uniform(-255, 255, content.shape).astype('float32')
    # Mix content with some noise
    image = noise * 0.3 + content * 0.7
    return image

def load_image(path):
    image = scipy.misc.imread(path).astype(np.float)
    # Reshape to add the extra dimension for the network
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Subtract the means
    image = image - RGB_MEANS
    return image

def save_image(path, image):
    # Add back the means
    image = image + RGB_MEANS
    # Drop the extra dimension
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

# TODO: make this more clear
def gram(input, n, m):
    # Reshape to 2D matrix
    matrix = tf.reshape(input, (n, m))
    return tf.matmul(tf.transpose(matrix), matrix)

# TODO: Clean up the loss functions
# Using squared error of orginal (F) and generated (P) as defined in paper
def content_loss(sess, model):

    loss = 0

    for layer in CONTENT_LAYERS:
        P = sess.run(model[layer])
        F = model[layer]
        loss += (1 / 2) * tf.reduce_sum(tf.pow(F - P, 2))

    return loss

def style_loss(sess, model):

    loss = 0

    for layer in STYLE_LAYERS:
        P = sess.run(model[layer])
        F = model[layer]

        # Number of filters
        N = P.shape[3]
        # Height x Width of feature map
        M = P.shape[1] * P.shape[2]

        # Gram matrix of generated image
        A = gram(P, N, M)
        # Gram matrix of style image
        G = gram(F, N, M)

        # TODO: change weights of layers?
        W = 0.2

        # E is the loss for a particular layer
        E = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2)) * W
        loss += E

    return loss

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

# TODO: make loop version of this
def create_graph():

    # Explicit step-by-step graph construction
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, HEIGHT, WIDTH, 3)), dtype = 'float32')

    graph['conv1_1']  = conv(graph['input'], 0)
    graph['relu1_1'] = relu(graph['conv1_1'])

    graph['conv1_2']  = conv(graph['relu1_1'], 2)
    graph['relu1_2'] = relu(graph['conv1_2'])

    graph['avgpool1'] = avgpool(graph['relu1_2'])

    graph['conv2_1']  = conv(graph['avgpool1'], 5)
    graph['relu2_1'] = relu(graph['conv2_1'])

    graph['conv2_2']  = conv(graph['relu2_1'], 7)
    graph['relu2_2'] = relu(graph['conv2_2'])

    graph['avgpool2'] = avgpool(graph['relu2_2'])

    graph['conv3_1']  = conv(graph['avgpool2'], 10)
    graph['relu3_1'] = relu(graph['conv3_1'])

    graph['conv3_2']  = conv(graph['relu3_1'], 12)
    graph['relu3_2'] = relu(graph['conv3_2'])

    graph['conv3_3']  = conv(graph['relu3_2'], 14)
    graph['relu3_3'] = relu(graph['conv3_3'])

    graph['conv3_4']  = conv(graph['relu3_3'], 16)
    graph['relu3_4'] = relu(graph['conv3_4'])

    graph['avgpool3'] = avgpool(graph['relu3_4'])

    graph['conv4_1']  = conv(graph['avgpool3'], 19)
    graph['relu4_1'] = relu(graph['conv4_1'])

    graph['conv4_2']  = conv(graph['relu4_1'], 21)
    graph['relu4_2'] = relu(graph['conv4_2'])

    graph['conv4_3']  = conv(graph['relu4_2'], 23)
    graph['relu4_3'] = relu(graph['conv4_3'])

    graph['conv4_4']  = conv(graph['relu4_3'], 25)
    graph['relu4_4'] = relu(graph['conv4_4'])

    graph['avgpool4'] = avgpool(graph['relu4_4'])

    graph['conv5_1']  = conv(graph['avgpool4'], 28)
    graph['relu5_1'] = relu(graph['conv5_1'])

    graph['conv5_2']  = conv(graph['relu5_1'], 30)
    graph['relu5_2'] = relu(graph['conv5_2'])

    graph['conv5_3']  = conv(graph['relu5_2'], 32)
    graph['relu5_3'] = relu(graph['conv5_3'])

    graph['conv5_4']  = conv(graph['relu5_3'], 34)
    graph['relu5_4'] = relu(graph['conv5_4'])

    graph['avgpool5'] = avgpool(graph['relu5_4'])

    return graph


if __name__ == '__main__':
    with tf.Session() as sess:

        # Load images
        content = load_image(CONTENT)
        style = load_image(STYLE)
        input = white_noise(content)

        # Create computation graph and initialize variables
        model = create_graph()
        sess.run(tf.global_variables_initializer())

        # Content and Style loss
        sess.run(model['input'].assign(content))
        L_content = content_loss(sess, model)

        sess.run(model['input'].assign(style))
        L_style = style_loss(sess, model)

        # Total loss
        L_total = BETA * L_content + ALPHA * L_style

        # Minimize loss
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(L_total)

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input))

        for it in range(ITERATIONS):
            sess.run(train_step)

        # Output final image and notify we're done
        output = sess.run(model['input'])
        filename = 'results/stylized.png'
        save_image(filename, output)
        print('Done.')
