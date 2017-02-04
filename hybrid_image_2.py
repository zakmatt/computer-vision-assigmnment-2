#!/usr/bin/env python3
import argparse
import cv2
import controllers
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

'''
def process_channels(image, sigma, is_low_pass, is_gray_scale):
    if is_gray_scale:
        image = process_image(image, sigma, is_low_pass)
    else:
        for channel in range(3):
            image[:, :, channel] = process_image(image[:, :, channel], sigma, is_low_pass)
        
    return image
        
    
def process_image(image, sigma, is_low_pass):
    n, m = image.shape
    image_fft = controllers.fourier_transform(image)
    kernel = controllers.GaussianKernel(n, m, sigma)
    image_filtered = controllers.filtering(image_fft, kernel)
    if not is_low_pass:
        image_filtered = image_fft - image_filtered
    image = controllers.revert_fourier_transform(image_filtered)
    image = np.real(image)#.astype('int')
    return image
'''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l",
                        "--low_freq_image",
                        help='Image that will be processed in low frequencies',
                        required=True)
    parser.add_argument("-hi",
                        "--high_freq_image",
                        help='Image that will be processed in high frequencies',
                        required=True)
    parser.add_argument("-g",
                        "--gray_scale",
                        help='Is image grayscale? T/F',
                        required=True)
    parser.add_argument("-k_1",
                        "--kernel_1_size",
                        help='Kernel 1 size',
                        required=True)
    parser.add_argument("-k_2",
                        "--kernel_2_size",
                        help='Kernel 2 size',
                        required=True)
    parser.add_argument("-s_1",
                        "--sigma_1",
                        help='Sigma',
                        required=True)
    parser.add_argument("-s_2",
                        "--sigma_2",
                        help='Sigma 2',
                        required=True)
    parser.add_argument("-p",
                        "--path",
                        help='Results save path',
                        required=False)
    
    args = parser.parse_args()
    
    low_image_path = args.low_freq_image
    high_image_path = args.high_freq_image
    is_grayscale = True if args.gray_scale == 'T' or args.gray_scale == 'True' else False
    kernel_1_size = int(args.kernel_1_size)
    kernel_2_size = int(args.kernel_2_size)
    sigma_1 = int(args.sigma_1)
    sigma_2 = int(args.sigma_2)
    path = args.path
    
    if kernel_1_size % 2 == 0 or kernel_2_size % 2 == 0:
        print('Kernel size has to be odd!')
        sys.exit()
    
    
    image_low = controllers.open_image(low_image_path, is_grayscale).astype('float32')
    image_high = controllers.open_image(high_image_path, is_grayscale).astype('float32')

    kernel_1 = controllers.GaussianKernel(kernel_1_size, sigma_1)
    kernel_2 = controllers.GaussianKernel(kernel_2_size, sigma_2)
    image_low = controllers.filtering(image_low, kernel_1)
    image_high -= controllers.filtering(image_high, kernel_2)
    output_image = controllers.rescale(image_low + image_high)

    cv2.imwrite('second_task_result.jpg', output_image.astype('uint8'))
    
    #comparizon
