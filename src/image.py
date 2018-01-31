import numpy as np
import scipy.misc

# VGG-19 mean RGB values
RGB_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def noisy_img(content):
    noise = np.random.uniform(-255, 255, content.shape).astype('float32')
    return image

def load_img(path):
    image = scipy.misc.imread(path).astype(np.float)
    # Reshape to add the extra dimension for the network
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Subtract the means
    image = image - RGB_MEANS
    return image

def save_img(path, image):
    # Add back the means
    image = image + RGB_MEANS
    # Drop the extra dimension
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
