import os
import numpy as np

from image import noisy_img
from image import load_img
from image import save_img
from model import create_model
from model import content_loss
from model import style_loss

import tensorflow as tf

ITERATIONS = 1000
LEARNING_RATE = 5.0
ALPHA = 10000
BETA = 1

STYLE = 'img/style/matisse.jpg'
CONTENT = 'img/content/sunflower.jpg'

if __name__ == '__main__':
    with tf.Session() as sess:

        # Load images
        content = load_img(CONTENT)
        style = load_img(STYLE)

        # Using content as base instead of white noise
        input = content

        model = create_model()

        # Content loss
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(content))
        L_content = content_loss(sess, model)

        # Style loss
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(style))
        L_style = style_loss(sess, model)

        # Total loss
        L_total = BETA * L_content + ALPHA * L_style
        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(L_total)

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input))

        for _ in range(ITERATIONS):
            sess.run(train_step)

        # Output the final image
        if not os.path.exists('img/results'):
            os.mkdir('img/results')
            
        output = sess.run(model['input'])
        filename = 'img/results/stylized_image.png'
        save_img(filename, output)
        print('Done.')
