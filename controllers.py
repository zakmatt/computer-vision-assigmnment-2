#!/usr/bin/env python3
import cv2
import numpy as np
from math import exp

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image

def GaussianKernel(size, sigma):
    centre = size // 2 + 1
    
    def get_coeffs(pos_x, pos_y):
        return exp(-1.0 * ((pos_x - centre)**2 +
                           (pos_y - centre)**2) / (2 * sigma**2))
    
    gaussian_filter = np.zeros((size, size))
    for pos_x in range(size):
        for pos_y in range(size):
            gaussian_filter[pos_x, pos_y] = get_coeffs(pos_x+1, pos_y+1)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

def filter_channel(channel, kernel):
    kernel_size = kernel.shape[0]
    padding_size = kernel_size // 2
    new_channel = np.lib.pad(channel, padding_size, 'constant', constant_values=0)
    new_size_x, new_size_y = new_channel.shape
    output_channel = np.zeros(channel.shape)
    for pos_x in range(new_size_x - kernel_size + 1):
        for pos_y in range(new_size_y - kernel_size + 1):
            cell_value = np.multiply(new_channel[pos_x:pos_x + kernel_size, pos_y:pos_y + kernel_size], kernel)
            cell_value = np.sum(cell_value)
            output_channel[pos_x, pos_y] = cell_value
    return output_channel
    
def filtering(image, kernel):
    image_shape = image.shape
    if len(image_shape) == 3:
        if image_shape[2] == 3:
            new_image = np.zeros(image_shape)
            for channel in range(3):
                new_image[:, :, channel] = filter_channel(image[:, :, channel], kernel)
        else:
            new_image = filter_channel(image, kernel)
    return new_image.astype('uint8')

def open_image(file_path, is_grayscale):
    return cv2.imread(file_path, 0) if is_grayscale else cv2.imread(file_path)
    
    
    
# high pass filter impulse filter - gaussian filter
    