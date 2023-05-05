#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    this program will blindly cut pictures same way
"""

import argparse
import glob
from time import sleep

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray


parser = argparse.ArgumentParser()
parser.add_argument('--thresh', default = 0.17, type = float, help = 'Thresh parametr')
parser.add_argument('--dir', default = '*/', type = str, help = 'directory name')
parser.add_argument('-e', default = '*', type = str, help = 'file type')
parser.add_argument('-x0', default = 0, type = int, help = 'minimal x')
parser.add_argument('-x1', default = 100, type = int, help = 'maximal x')
parser.add_argument('-y0', default = 0, type = int, help = 'minimal y')
parser.add_argument('-y1', default = 100, type = int, help = 'maximal y')
parser.add_argument('--ex', default = 'tif', type = str, help = 'file type export')



def simple_cut(args, file_name):
    """
        cut and save file
    """
    image = io.imread(file_name)
    mini_image = image[args.y0:args.y1,args.x0:args.x1,:]
    io.imsave(file_name.replace(f'.{args.e}', f'_cut.{args.ex}'), mini_image)

def separate(args, file_name):
    """"""
    image = io.imread(file_name)
    image_gray = rgb2gray(image)

    index = 0

    thresh = args.thresh
    # print('\t', '\t', thresh)
    bw = closing(image_gray > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image_gray, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 50000:
            if (region.bbox[2]-region.bbox[0])*(region.bbox[3]-region.bbox[1]) >= 1500000:
                print('\t', '\t', region.area, region.bbox, (region.bbox[2]-region.bbox[0])*(region.bbox[3]-region.bbox[1]))
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

                mini_image = image[minr:maxr,minc:maxc,:]
                io.imsave(file_name.replace('.jpg', f'_{index}.tif'), mini_image)
                mini_image = ''
                index += 1

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(file_name.replace('.jpg', '.png'))
    sleep(5)
    plt.close()

def main(args):
    """
        find all pictures in directory and process them
    """
    print(args.dir)
    for file_name in glob.glob(f'{args.dir}*.{args.e}'):
        print('\t', file_name)
        simple_cut(args, file_name)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
