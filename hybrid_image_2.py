#!/usr/bin/env python3
import argparse
import cv2
import controllers
import matplotlib.pyplot as plt
import numpy as np
import os

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
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-l",
                        "--low_freq_image",
                        help='Image that will be processed in low frequencies',
                        required=True)
    parser.add_argument("-hf",
                        "--high_freq_image",
                        help='Image that will be processed in high frequencies',
                        required=True)
    parser.add_argument("-s",
                        "--sigma",
                        help='Sigma in kernel',
                        required=True)
    parser.add_argument("-p",
                        "--save_path",
                        help='Path to save the result',
                        required=False)
    parser.add_argument("-g",
                        "--is_greyscale",
                        help='Are images grayscale or not',
                        required=True)
    
    args = parser.parse_args()
    image_low_path = args.low_freq_image
    image_high_path = args.high_freq_image
    sigma = int(args.sigma)
    save_path = args.save_path
    is_grayscale = args.is_greyscale == 'True'
    image_low = controllers.open_image(image_low_path, is_grayscale)
    image_high = controllers.open_image(image_high_path, is_grayscale)
    '''
    
    image_low = controllers.open_image('data/dog.bmp', False)
    image_high = controllers.open_image('data/cat.bmp', False)
    kernel = controllers.GaussianKernel(11, 7)
    '''
    kernel_2 = cv2.getGaussianKernel(5, 3)
    kernel_2 = kernel_2 * kernel_2.T
    '''
    image_low = controllers.filtering(image_low, kernel)
    image_high -= controllers.filtering(image_high, kernel)
    plt.imshow(image_low + image_high)
    plt.show()
    cv2.imwrite('tmp.jpg', (image_low + image_high)/2)
    
    
    '''
    tmp = image_low[:, :, 0]
    tmp_img = image[:, :, 0]
    output = tmp_img.astype('uint8')
    output_image = image_low.astype('uint8')
    plt.imshow(output_image)
    plt.show()
    '''
    '''
    image_low = process_channels(image_low, sigma, True, is_grayscale).astype('float32')
    image_high = process_channels(image_high, sigma, False, is_grayscale).astype('float32')
    image = image_low + image_high
    #image = cast_values(image)
    image = image.astype('uint8')
    if save_path is None:
        save_path = ''
        
    save_path = os.path.join(save_path, 'result_task_2.jpeg')
    cv2.imwrite(save_path, image)
    '''