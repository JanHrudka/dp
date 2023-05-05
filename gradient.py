#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
"""
import argparse
import glob

from PIL import Image, ImageDraw
import numpy as np
from skimage.filters import sobel_h, sobel_v

parser = argparse.ArgumentParser()
parser.add_argument('--zone_count', default = 2, type = int, help = 'How many zones?')
parser.add_argument('--model_dimension', default = 3, type = int, help = 'How many model dimensions?')
parser.add_argument('--sample_file_name', default = '', type = str, help = 'Sample file name?')
parser.add_argument('--test_file_name', default = '', type = str, help = 'Test file name?')
parser.add_argument('--edge_parameter', default = 0.12, type = float, help = 'Edge parametr')
parser.add_argument('--gauss_blur_parametr', default = 0.5, type = float, help = 'gauss blur parametr')
parser.add_argument('--do_new_model', default = False, type = bool, help = 'train new model?')
parser.add_argument('--dir', default = '*/', type = str, help = 'directory name')
parser.add_argument('-e', default = '_cut.tif', type = str, help = 'file type')
parser.add_argument('-r', default = .25, type = float, help = 'rescale')

parser.add_argument('-t', default = 165, type = float, help = 'threshold')


def main(args):
    """"""
    for file_name in glob.glob(f'{args.dir}*{args.e}'):
        print('\t', file_name)

        # Load the image and convert it to a 2D array of pixel values
        image = Image.open(file_name)
        pixels = np.array(image)

        # Calculate the center and radius of the circle
        center_x, center_y = int(pixels.shape[0] / 2), int(pixels.shape[1] / 2)
        radius = pixels.shape[0] / 2.4

        # # Cut out everything outside of the circle
        # for x in range(pixels.shape[0]):
        #     for y in range(pixels.shape[1]):
        #         if (x - center_x)**2 + (y - center_y)**2 > radius**2:
        #             pixels[x, y] = (0, 0, 0)

        # Select the red channel of the image
        red_channel = pixels[:,:,0]

        # Calculate the horizontal and vertical gradients of the image
        horizontal_gradient = sobel_h(red_channel)
        vertical_gradient = sobel_v(red_channel)

        # Calculate the gradient magnitude using the horizontal and vertical gradients
        gradient_magnitude = np.sqrt(horizontal_gradient**2 + vertical_gradient**2)

        # Create a new image with the same size as the original image
        separated_image = np.zeros_like(pixels)

        # Separate the pixels into two zones using the gradient magnitude as a threshold
        light_pixels = gradient_magnitude > args.t  # Boolean mask for light pixels
        separated_image[light_pixels] = (255, 255, 255)  # Set light pixels to white
        separated_image[~light_pixels] = (0, 0, 0)  # Set rest of the pixels to black

        # Convert the separated image to a PIL Image object and display it
        separated_image = Image.fromarray(separated_image.astype(np.uint8))
        # separated_image.show()
        separated_image.save(file_name.replace('.tif', '_t.tif'))

        # Scale the gradient magnitude array to a range of 0 to 255
        scaled_gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min()) * 255

        # Convert the gradient magnitude array to an 8-bit integer type
        gradient_magnitude_image = scaled_gradient_magnitude.astype(np.uint8)

        # Convert the gradient magnitude array to a PIL Image object and display it
        gradient_magnitude_image = Image.fromarray(gradient_magnitude_image)
        gradient_magnitude_image.save(file_name.replace('.tif', '_g.tif'))



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
