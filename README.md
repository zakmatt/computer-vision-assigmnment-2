# Hybrid Images  
## Usage:  
`python hybrid_imageXX.py -l image -hi image -g is grayscale -k_1 kernel 1 size -k_2 kernel 2 size -s_1 sigma 1 -s_2 sigma 2 -p save path`  
`-l image` - input image that will be processed in low frequencies  
`-hi image` - input image that will be processed in high frequencies  
`-g is grayscale` - 'T' or 'F' signifying whether output images are grayscale or not  
`-k_1 kernel 1 size` - kernel size for processing an image in low frequencies  
`-k_2 kernel 2 size` - kernel size for processin an image in high frequencies  
`-s_1 sigma` - sigma parameter for processing an image in low frequencies  
`-s_2 sigma` - sigma parameter for processing an image in high frequencies  
  
`XX` corresponds to the number of task we would like to run. When XX is skipped, `hybrid_image.py` is run. When XX equals _2 or _3 then we run `hybrid_image_2` and `hybrid_image_3.py`, respectively.  
`hybrid_image_3.py` does not accept any arguments as inputs.