#!/usr/bin/env python3
import cv2
import controllers
import numpy as np
import matplotlib.pyplot as plt

def generate_oryginal_image(image_low, image_high, size, sigma):
    image_low = controllers.open_image(image_low, False).astype('float32')
    image_high = controllers.open_image(image_high, False).astype('float32')

    oryginal_low = cv2.GaussianBlur(image_low, (size, size), 33).astype('float32')
    oryginal_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, (size, size), 17).astype('float32')
    oryginal_output = controllers.rescale(oryginal_low + oryginal_high)
    return oryginal_output

def sobel_test(image_low, image_high, size, sigma):
    kernel = controllers.GaussianKernel(size, sigma)
    image_low = controllers.open_image(image_low, False).astype('float32')
    image_high = controllers.open_image(image_high, False).astype('float32')
    
    image_low = controllers.filtering(image_low, kernel)
    image_low = controllers.rescale(image_low)
    
    sobel_high = controllers.sobel_filtering(image_high)
    sobel_high = controllers.rescale(sobel_high)
    sobel_output = controllers.rescale(image_low + sobel_high)
    cv2.imwrite('results/sobel/sobel_%d.jpeg' % sigma, sobel_output.astype('uint8'))
    
    oryginal_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, (size, size), 17).astype('float32')
    oryginal_high = controllers.rescale(oryginal_high)
    oryginal = controllers.rescale(image_low + oryginal_high).astype('float32')
    comparison = np.sqrt((oryginal - sobel_output) ** 2)
    comparison = controllers.rescale(comparison)
    cv2.imwrite('results/sobel/sobel_comparison_%d.jpeg' % sigma, comparison.astype('uint8'))
    
def dog_test(image_low, image_high, size, sigma_1, sigma_2):
    image_low = controllers.open_image(image_low, False).astype('float32')
    image_high = controllers.open_image(image_high, False).astype('float32')

    image_low = cv2.GaussianBlur(image_low, (size, size), 20).astype('float32')
    oryginal_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, (size, size), 17).astype('float32')
    oryginal_output = controllers.rescale(image_low + oryginal_high).astype('float32')
    
    d_o_g_filter = controllers.d_o_g(size, sigma_1, sigma_2)
    image_high = controllers.filtering(image_high, d_o_g_filter)
    image_high = controllers.rescale(image_high)
    dog_output = controllers.rescale(image_low + image_high).astype('float32')
    cv2.imwrite('results/dog/dog_%d_%d_%d.jpeg' % (size, sigma_1, sigma_2), dog_output.astype('uint8'))
    
    comparison = np.sqrt((oryginal_output - dog_output) ** 2)
    comparison = controllers.rescale(comparison)
    cv2.imwrite('results/dog/dog_comparison_%d_%d_%d.jpeg' % (size, sigma_1, sigma_2), comparison.astype('uint8'))
    
def laplacian_test(image_low, image_high, size, sigma):
    image_low = controllers.open_image(image_low, False).astype('float32')
    image_high = controllers.open_image(image_high, False).astype('float32')
    
    image_low = cv2.GaussianBlur(image_low, (size, size), 20).astype('float32')
    image_low = controllers.rescale(image_low)
    oryginal_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, (size, size), 17).astype('float32')
    oryginal_output = controllers.rescale(image_low + oryginal_high).astype('float32')
    
    kernel_l_o_g = controllers.l_o_g_kernel(size, sigma)
    image_high = controllers.filtering(image_high, kernel_l_o_g)
    image_high = controllers.rescale(image_high)
    log_output = controllers.rescale(image_low + image_high).astype('float32')
    cv2.imwrite('results/laplacian/log_%d_%d.jpeg' % (size, sigma), log_output.astype('uint8'))
    
    comparison = np.sqrt((oryginal_output - log_output) ** 2)
    comparison = controllers.rescale(comparison)
    cv2.imwrite('results/laplacian/log_comparison_%d_%d.jpeg' % (size, sigma), comparison.astype('uint8'))
    
if __name__ == '__main__':
    image_low = 'data/dog.bmp'
    image_high = 'data/cat.bmp'
    # Sobel 
    for i in range(20, 80, 20):
        sobel_test(image_low, image_high, 33, i)
        
    # DoG
    dog_test(image_low, image_high, 33, 10, 30)
    dog_test(image_low, image_high, 33, 30, 10)
    dog_test(image_low, image_high, 11, 10, 30)
    dog_test(image_low, image_high, 11, 30, 10)
    
    # LoG
    laplacian_test(image_low, image_high, 21, 10)
    laplacian_test(image_low, image_high, 21, 25)
    laplacian_test(image_low, image_high, 21, 40)
    laplacian_test(image_low, image_high, 21, 10)
    laplacian_test(image_low, image_high, 31, 25)
    laplacian_test(image_low, image_high, 41, 40)
