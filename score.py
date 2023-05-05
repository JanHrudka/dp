#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    comparison of results to find the best one
"""

import argparse
import glob

from PIL import Image
import numpy as np
import cv2 as cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default = '*/', type = str, help = 'directory name')
parser.add_argument('-e', default = '_cut.tif', type = str, help = 'file type')
parser.add_argument('-e2', default = '.TIF', type = str, help = 'file type')


parser.add_argument('-c', default = 2.3, type = float, help = 'circular cut radius')
parser.add_argument('-a', default = 2001, type = int, help = 'area of adaptive zone')
parser.add_argument('-t', default = 16, type = int, help = 'threshold')
parser.add_argument('--st', default = 100, type = int, help = 'threshold')

# parser.add_argument('-t', default = 165, type = float, help = 'threshold')

def normalize(pixels):
    '''
        normalize image size so it is squire
            update now it can take different shapes
    '''
    x = pixels.shape[0]
    y = pixels.shape[1]

    if len(pixels.shape) == 3:
        if x > y:
            pixels = pixels[:y,:,:]
        else:
            pixels = pixels[:,:x,:]
    elif len(pixels.shape) == 2:
        if x > y:
            pixels = pixels[:y,:]
        else:
            pixels = pixels[:,:x]
    return pixels

def prepare(file_name, c):
    '''
        prepare variables
    '''
    # Load the image and convert it to a 2D array of pixel values
    image = Image.open(file_name)
    pixels = np.array(image)
    pixels = normalize(pixels)

    # Calculate the center and radius of the circle
    center_x, center_y = int(pixels.shape[0] / 2), int(pixels.shape[1] / 2)
    radius = pixels.shape[0] / c
    x = np.arange(0, pixels.shape[0])
    y = np.arange(0, pixels.shape[1])
    mask = (x[np.newaxis,:]-center_x)**2 + (y[:,np.newaxis]-center_y)**2 > radius**2

    # stats
    total_area = pixels.shape[0]*pixels.shape[1]
    circular_cut_of_area = np.sum(mask)

    return pixels, mask, total_area, circular_cut_of_area

def simple_threshold(file_name, pixels, mask, threshold):
    '''
        do a simple threshold
    '''

    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    gray[mask] = 0
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    gray_threshold = blurred[:,:] > threshold
    gray[gray_threshold] = 255
    gray[~gray_threshold] = 0

    # Convert the separated image to a PIL Image object and display it
    separated_image = Image.fromarray(gray.astype(np.uint8))
    separated_image.save(file_name.replace('.tif', '_st.TIF'))

    return gray

def adapted_recognition(file_name, pixels, mask, inverted, a, t):
    '''
        do adapted recognition
    '''
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    if inverted:
        gray = 255 - gray
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    adap_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, a, t)
    if inverted:
        adap_thresh = 255 - adap_thresh

    # circular cut of
    adap_thresh[mask] = 0

    # Convert the separated image to a PIL Image object and display it
    separated_image = Image.fromarray(adap_thresh.astype(np.uint8))
    if inverted:
        separated_image.save(file_name.replace('.tif', '_i.TIF'))
    else:
        separated_image.save(file_name.replace('.tif', '_n.TIF'))

    return adap_thresh

def get_score():
    '''
        gets final score
    '''

def stats(image, total_area, circular_cut_of_area):
    '''
        get stats
    '''
    zone_area = np.sum(image//255)
    print(zone_area/(total_area-circular_cut_of_area), end=' ')

    # stats of top
    zone_area = np.sum(image[:image.shape[0]//2, :]//255)
    print(zone_area/(total_area/2-circular_cut_of_area/2), end=' ')

    # stats of bot
    zone_area = np.sum(image[image.shape[0]//2:, :]//255)
    print(zone_area/(total_area/2-circular_cut_of_area/2), end=' ')

def main(args):
    """"""
    c = args.c
    # a = args.a
    t = args.t

    # print(c, a, t)
    # print('file_name', 'simple', 'simple_top', 'simple_bot', 'inverted', 'inverted_top', 'inverted_bot', 'normal', 'normal_top','normal_bot')

    for file_name in sorted(glob.glob(f'{args.dir}*{args.e}')):
        print(file_name.split('/')[-1].split('.')[0])
        name_precursor = '..' + file_name.split('.')[-2] + '_'

        pixels, mask, total_area, circular_cut_of_area = prepare(file_name, c)

        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        gray[mask] = 0


        for file_name_to_compare in sorted(glob.glob(f'{name_precursor}*{args.e2}')):
            print('\t', file_name_to_compare)
            pixels_to_compare, mask_to_compare, total_area_to_compare, circular_cut_of_area_to_compare = prepare(file_name_to_compare, c)



            truly_positive = gray[pixels_to_compare]


            truly_positive_threshold = truly_positive[:,:] > t
            truly_positive[truly_positive_threshold] = 1
            truly_positive[~truly_positive_threshold] = -1

            truly_positive[mask] = 0

            print('\t\t', np.sum(truly_positive))


            falsy_negative = gray[~pixels_to_compare]
            falsy_negative[mask] = 0


            falsy_negative_threshold = falsy_negative[:,:] > t
            falsy_negative[falsy_negative_threshold] = -1
            falsy_negative[~falsy_negative_threshold] = 2

            falsy_negative[mask] = 0

            print('\t\t', np.sum(falsy_negative))

            print('\t\t', np.sum(truly_positive) + np.sum(falsy_negative))







        # ## simple threshold
        # gray = simple_threshold(file_name, pixels, mask, args.st)

        # ## stats
        # stats(gray, total_area, circular_cut_of_area)

        # ## inverted adapted recognition
        # adap_thresh = adapted_recognition(file_name, pixels, mask, True, a, t)

        # ## stats
        # stats(adap_thresh, total_area, circular_cut_of_area)

        # ## normal adapted recognition
        # adap_thresh = adapted_recognition(file_name, pixels, mask, False, a, t)

        # ## stats
        # stats(adap_thresh, total_area, circular_cut_of_area)

        # print()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
