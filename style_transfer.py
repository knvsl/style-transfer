import os
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

VGG = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
ITERATIONS = 1000
LEARNING_RATE = 3.0
ALPHA = 100
BETA = 2

CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

STYLE = 'img/style/vangogh.jpg'
CONTENT = 'img/content/sunflower.jpg'
WIDTH = 800
HEIGHT = 600

# VGG-19 mean RGB
RGB_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

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

def gram(input, n, m):
    # Reshape to 2D matrix
    matrix = tf.reshape(input, (m, n))
    return tf.matmul(tf.transpose(matrix), matrix)

# Using squared error of generated (F) and original (P) as defined in paper
def content_loss(sess, model):

    loss = 0

    for layer in CONTENT_LAYERS:
        F = sess.run(model[layer])
        P = model[layer]
        N = F.shape[3]
        M = F.shape[1] * F.shape[2]
        loss += (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(F - P, 2))

    return loss

def style_loss(sess, model):

    loss = 0

    for layer in STYLE_LAYERS:
        F = sess.run(model[layer])
        P = model[layer]

        # Number of filters
        N = F.shape[3]
        # Height x Width of feature map
        M = F.shape[1] * F.shape[2]
        # Gram matrix of original image
        A = gram(P, N, M)
        # Gram matrix of generated image
        G = gram(F, N, M)

        W = 0.2

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

    # Initial input
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

"""
def create_model():

    model = {}
    model['input']   = tf.Variable(np.zeros((1, HEIGHT, WIDTH, 3)), dtype = 'float32')

    model['conv1_1']  = conv(model['input'], 0)
    model['relu1_1'] = relu(model['conv1_1'])

    model['conv1_2']  = conv(model['relu1_1'], 2)
    model['relu1_2'] = relu(model['conv1_2'])

    model['avgpool1'] = avgpool(model['relu1_2'])

    model['conv2_1']  = conv(model['avgpool1'], 5)
    model['relu2_1'] = relu(model['conv2_1'])

    model['conv2_2']  = conv(model['relu2_1'], 7)
    model['relu2_2'] = relu(model['conv2_2'])

    model['avgpool2'] = avgpool(model['relu2_2'])

    model['conv3_1']  = conv(model['avgpool2'], 10)
    model['relu3_1'] = relu(model['conv3_1'])

    model['conv3_2']  = conv(model['relu3_1'], 12)
    model['relu3_2'] = relu(model['conv3_2'])

    model['conv3_3']  = conv(model['relu3_2'], 14)
    model['relu3_3'] = relu(model['conv3_3'])

    model['conv3_4']  = conv(model['relu3_3'], 16)
    model['relu3_4'] = relu(model['conv3_4'])

    model['avgpool3'] = avgpool(model['relu3_4'])

    model['conv4_1']  = conv(model['avgpool3'], 19)
    model['relu4_1'] = relu(model['conv4_1'])

    model['conv4_2']  = conv(model['relu4_1'], 21)
    model['relu4_2'] = relu(model['conv4_2'])

    model['conv4_3']  = conv(model['relu4_2'], 23)
    model['relu4_3'] = relu(model['conv4_3'])

    model['conv4_4']  = conv(model['relu4_3'], 25)
    model['relu4_4'] = relu(model['conv4_4'])

    model['avgpool4'] = avgpool(model['relu4_4'])

    model['conv5_1']  = conv(model['avgpool4'], 28)
    model['relu5_1'] = relu(model['conv5_1'])

    model['conv5_2']  = conv(model['relu5_1'], 30)
    model['relu5_2'] = relu(model['conv5_2'])

    model['conv5_3']  = conv(model['relu5_2'], 32)
    model['relu5_3'] = relu(model['conv5_3'])

    model['conv5_4']  = conv(model['relu5_3'], 34)
    model['relu5_4'] = relu(model['conv5_4'])

    model['avgpool5'] = avgpool(model['relu5_4'])

    return model
"""

if __name__ == '__main__':
    with tf.Session() as sess:

        # Load images
        content = load_image(CONTENT)
        style = load_image(STYLE)
        input = white_noise(content)

        # Create computation model and initialize variables
        model = create_model()


        # Content and Style loss
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(content))
        L_content = content_loss(sess, model)

        sess.run(tf.global_variables_initializer())
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

            # Print stats for testing
            if it%100 == 0:
                # Print every 100 iteration.
                mixed_image = sess.run(model['input'])
                print('Iteration %d' % (it))

                filename = 'results/%d.png' % (it)
                save_image(filename, mixed_image)

        # Output final image and notify we're done
        output = sess.run(model['input'])
        filename = 'results/stylized.png'
        save_image(filename, output)
        print('Done.')
