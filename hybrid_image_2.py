#!/usr/bin/env python3
import argparse
import cv2
import controllers
import os
import numpy as np
import sys

def inbuilt_filtering(image_low, image_high, kernel_1_size, kernel_2_size, sigma_1, sigma_2):
    kernel_1 = (kernel_1_size, kernel_1_size)
    kernel_2 = (kernel_2_size, kernel_2_size)
    image_low = controllers.open_image(low_image_path, is_grayscale)
    image_high = controllers.open_image(high_image_path, is_grayscale)
    
    image_low = cv2.GaussianBlur(image_low, kernel_1, 0).astype('float32')
    image_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, kernel_2, 0).astype('float32')

    output_image = image_low+image_high
    
    return output_image
    
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

    path = '' if path == None else path
    
    cv2.imwrite(os.path.join(path, 'second_task_result.jpg'), controllers.rescale(output_image).astype('uint8'))
    
    #comparizon
    inbuilt_filtered = inbuilt_filtering(low_image_path, high_image_path, kernel_1_size, kernel_2_size, sigma_1, sigma_2)
    comparison = np.sqrt((output_image - inbuilt_filtered) ** 2)
    comparison = controllers.rescale(comparison).astype('uint8')
    cv2.imwrite(os.path.join(path, 'comparison.jpg'), comparison)
    
