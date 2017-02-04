#!/usr/bin/env python3
import argparse
from controllers import open_image, rescale
import cv2
import os
import sys

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
                        required=False)
    parser.add_argument("-s_2",
                        "--sigma_2",
                        help='Sigma 2',
                        required=False)
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
    sigma_1 = 0 if args.sigma_1 == None else int(args.sigma_1)
    sigma_2 = 0 if args.sigma_2 == None else int(args.sigma_2)
    path = args.path
    
    if kernel_1_size % 2 == 0 or kernel_2_size % 2 == 0:
        print('Kernel size has to be odd!')
        sys.exit()
    
    kernel_1 = (kernel_1_size, kernel_1_size)
    kernel_2 = (kernel_2_size, kernel_2_size)
    image_low = open_image(low_image_path, is_grayscale)
    image_high = open_image(high_image_path, is_grayscale)
    
    image_low = cv2.GaussianBlur(image_low, kernel_1, 0).astype('float32')
    image_high = image_high.astype('float32') - cv2.GaussianBlur(image_high, kernel_2, 0).astype('float32')

    output_image = image_low+image_high
    # rescale to 255
    output_image = rescale(output_image).astype('uint8')
    
    path = '' if path == None else path
    cv2.imwrite(os.path.join(path, 'first_task_result.jpg'), output_image)
    