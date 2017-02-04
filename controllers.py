#!/usr/bin/env python3
import cv2
import numpy as np
from math import exp
from numpy.fft import fft2, ifft2, fftshift, ifftshift
'''
def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
 
   def gaussian(i,j):
      coefficient = exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient
 
   return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])
'''
def GaussianKernel(sizex, sizey, sigma):
    if sizex % 2 == 1:
        centre_x = int(sizex / 2 + 1)
    else:
        centre_x = int(sizex / 2)
    if sizey % 2 == 1:
        centre_y = int(sizey / 2 + 1)
    else:
        centre_y = int(sizey / 2)
        
    def get_coffs(pos_x, pos_y):
        return exp(-1.0 * ((pos_x - centre_x)**2 +
                           (pos_y - centre_y)**2) / (2 * sigma**2))
    filter = np.zeros((sizex, sizey))
    for pos_x in range(sizex):
        for pos_y in range(sizey):
            filter[pos_x, pos_y] = get_coffs(pos_x, pos_y)
    
    return filter
   
def transform_all_channels(image, kernel):
    _, _, num_of_channels = image.shape
    for channel in range(num_of_channels):
        image[:, :, channel] *= kernel
    return image
    
def filtering(image, kernel):
    image_real = np.real(np.asanyarray(image.real, dtype = np.float32))
    image_imag = np.real(np.asanyarray(image.imag, dtype = np.float32))
    image_real *= kernel
    image_imag *= kernel
    return np.round(image_real) + 1j * np.round(image_imag)
    
def open_image(file_path, is_grayscale):
    return cv2.imread(file_path, 0) if is_grayscale else cv2.imread(file_path)
    
def fourier_transform(image):
    # frequency transform (complex array)
    return fftshift(fft2(image))

def revert_fourier_transform(image):
    return ifft2(ifftshift(image))
    
# low pass filter standard 2D gaussian filter
def low_pass_filtering(image):
    
    return cv2.GaussianBlur(np.asarray(image), (5,5), 0)
    
    
# high pass filter impulse filter - gaussian filter
    