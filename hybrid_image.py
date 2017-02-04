#!/usr/bin/env python3
import argparse
from controllers import open_image
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
    parser.add_argument("-k",
                        "--kernel_size",
                        help='Kernel size',
                        required=True)
    parser.add_argument("-s",
                        "--sigma",
                        help='Sigma',
                        required=True)
    parser.add_argument("-p",
                        "--path",
                        help='Results save path',
                        required=False)
    
    args = parser.parse_args()
    
    low_image_path = args.low_freq_image
    high_image_path = args.high_freq_image
    is_grayscale = True if args.gray_scale == 'T' or args.gray_scale == 'True' else False
    kernel_size = int(args.kernel_size)
    sigma = int(args.sigma)
    path = args.path
    
    if kernel_size % 2 == 0:
        print('Kernel size has to be odd!')
        sys.exit()
    
    kernel = (kernel_size, kernel_size)
    image_low = open_image(low_image_path, is_grayscale)
    image_high = open_image(high_image_path, is_grayscale)
    
    image_low = cv2.GaussianBlur(image_low, kernel, sigma)
    image_high -= cv2.GaussianBlur(image_high, kernel, sigma)
    image_high[image_high < 50] = 0
    
    path = '' if path == None else path
    cv2.imwrite(os.path.join(path, 'first_task_result.jpg'), image_low+image_high)
    