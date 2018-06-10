import imageio
import numpy as np

#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path, range_min=-1.0, range_max=1.0):
    return imsave(inverse_transform(images, range_min, range_max), size, image_path)

def inverse_transform(images, range_min, range_max):
    return (images - range_min)/(range_max - range_min)

def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))

def merge(images, size):
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
