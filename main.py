#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    image   --> bitmap  --> preprocessed bitmap     --> processed bitmap    --> image
                                                                            --> statistic   --> data    --> plots
"""

import glob

import argparse
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.segmentation import slic

from skimage import filters

from skimage.feature import blob_log
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imshow

from skimage.morphology import erosion
from skimage.color import rgb2gray

import itertools
import random

import tensorflow as tf
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

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


def split_dataset(X, Y, train, test):
    """ split dataset to numpy array train, validation and test sets """
    train = int(len(X)*train)
    test = int(len(X)*test) + train

    X_train = np.asarray(X[0:train])
    X_test = np.asarray(X[train:test])
    X_validation = np.asarray(X[test:])

    Y_train = np.asarray(Y[0:train], dtype=np.float16)
    Y_test = np.asarray(Y[train:test], dtype=np.float16)
    Y_validation = np.asarray(Y[test:], dtype=np.float16)

    return X_train, X_test, X_validation, Y_train, Y_test, Y_validation

def get_dataset(enriched_bitmap, final_bitmap):
    """ create dataset from enriched bitmap and result bitmap"""
    edge_parameter = 0.12

    X = []
    Y = []

    for i, row_enriched_pixels in enumerate(enriched_bitmap):
        for j, enriched_pixel in enumerate(row_enriched_pixels):
            if enriched_pixel[9] > edge_parameter:
                X.append([enriched_pixel[10], enriched_pixel[11], enriched_pixel[12]])
                if final_bitmap[i][j] == [255, 0, 0]:
                    Y.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif final_bitmap[i][j] == [0, 255, 0]:
                    Y.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif final_bitmap[i][j] == [0, 0, 255]:
                    Y.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                elif final_bitmap[i][j] == [255, 0, 255]:
                    Y.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                elif final_bitmap[i][j] == [0, 255, 255]:
                    Y.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                elif final_bitmap[i][j] == [255, 255, 0]:
                    Y.append([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                if final_bitmap[i][j] == [0, 0, 0]:
                    Y.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    print(len(X), len(Y))
    temp = list(zip(X, Y))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    X, Y = list(res1), list(res2)

    return X, Y

def use_KM(enriched_bitmap, edge_parameter, model, model_dimension):
    """ use learned model """
    bitmap = []
    nice_bitmap = []
    relative_color_bitmap = condensate_enriched_bitmap_for_KM(enriched_bitmap, edge_parameter, model_dimension)
    model_result = model.predict(relative_color_bitmap)

    index_zones = 0
    for row_enriched_pixels in enriched_bitmap:
        row = []
        nice_row = []
        for enriched_pixel in row_enriched_pixels:
            if enriched_pixel[9] > edge_parameter:
                if model_result[index_zones] == 0:
                    nice_row.append([enriched_pixel[0], 0, 0])
                    row.append([255, 0, 0])
                elif model_result[index_zones] == 1:
                    nice_row.append([0, enriched_pixel[1], 0])
                    row.append([0, 255, 0])
                elif model_result[index_zones] == 2:
                    nice_row.append([0, 0, enriched_pixel[2]])
                    row.append([0, 0, 255])
                elif model_result[index_zones] == 3:
                    nice_row.append([enriched_pixel[2], 0, enriched_pixel[2]])
                    row.append([255, 0, 255])
                elif model_result[index_zones] == 4:
                    nice_row.append([0, enriched_pixel[1], enriched_pixel[2]])
                    row.append([0, 255, 255])
                elif model_result[index_zones] == 5:
                    nice_row.append([enriched_pixel[0], enriched_pixel[1], 0])
                    row.append([255, 255, 0])
                index_zones += 1
            else:
                nice_row.append([0, 0, 0])
                row.append([0, 0, 0])
        bitmap.append(row)
        nice_bitmap.append(nice_row)

    return bitmap, nice_bitmap

def condensate_enriched_bitmap_for_KM(enriched_bitmap, edge_parameter, model_dimension):
    """ relative colors of pixels in dish """
    relative_color_bitmap = []
    if model_dimension == 3:
        for i, row_enriched_pixels in enumerate(enriched_bitmap):
            for j, enriched_pixel in enumerate(row_enriched_pixels):
                if enriched_pixel[9] > edge_parameter:
                    relative_color_bitmap.append([enriched_pixel[10], enriched_pixel[11], enriched_pixel[12]])
    elif model_dimension == 4:
        for i, row_enriched_pixels in enumerate(enriched_bitmap):
            for j, enriched_pixel in enumerate(row_enriched_pixels):
                if enriched_pixel[9] > edge_parameter:
                    relative_color_bitmap.append([enriched_pixel[10], enriched_pixel[11], enriched_pixel[12], enriched_pixel[9]])
    else:
        exit(0)
    return relative_color_bitmap

def processed_bitmap_by_KM(enriched_bitmap, edge_parameter, zone_count, initial_points_relative_colors, model_dimension):
    """ evaluate enriched bitmap to basic bitmap by ML """
    score = [0,0,0,0,0,0,0,0]
    bitmap = []
    nice_bitmap = []
    model = KMeans(n_init=10, n_clusters=zone_count, init=np.asanyarray(initial_points_relative_colors))
    # model = KMeans(n_clusters=zone_count)
    # model = KMeans(n_clusters=zone_count)
    relative_color_bitmap = condensate_enriched_bitmap_for_KM(enriched_bitmap, edge_parameter, model_dimension)
    model.fit(relative_color_bitmap)
    # print(model.cluster_centers_)

    # print(model.predict([[0.0, 0.0, 0.0]]))

    # connect solution with enriched bitmap
    index_zones = 0
    for row_enriched_pixels in enriched_bitmap:
        row = []
        nice_row = []
        for enriched_pixel in row_enriched_pixels:
            if enriched_pixel[9] > edge_parameter:
                if model.labels_[index_zones] == 0:
                    nice_row.append([enriched_pixel[0], 0, 0])
                    row.append([255, 0, 0])
                    score[model.labels_[index_zones]] += 1
                elif model.labels_[index_zones] == 1:
                    nice_row.append([0, enriched_pixel[1], 0])
                    row.append([0, 255, 0])
                    score[model.labels_[index_zones]] += 1
                elif model.labels_[index_zones] == 2:
                    nice_row.append([0, 0, enriched_pixel[2]])
                    row.append([0, 0, 255])
                    score[model.labels_[index_zones]] += 1
                elif model.labels_[index_zones] == 3:
                    nice_row.append([enriched_pixel[0], 0, enriched_pixel[2]])
                    row.append([255, 0, 255])
                    score[model.labels_[index_zones]] += 1
                elif model.labels_[index_zones] == 4:
                    nice_row.append([0, enriched_pixel[1], enriched_pixel[2]])
                    row.append([0, 255, 255])
                    score[model.labels_[index_zones]] += 1
                elif model.labels_[index_zones] == 5:
                    nice_row.append([enriched_pixel[0], enriched_pixel[1], 0])
                    row.append([255, 255, 0])
                    score[model.labels_[index_zones]] += 1
                index_zones += 1
            else:
                nice_row.append([0, 0, 0])
                row.append([0, 0, 0])
        bitmap.append(row)
        nice_bitmap.append(nice_row)

    return bitmap, nice_bitmap, model, score

def get_processed_bitmap_old(enriched_bitmap):
    """ evaluate enriched bitmap to basic bitmap """
    edge_parameter = 0.12
    threshold = 0.8

    bitmap = []
    for row_enriched_pixels in enriched_bitmap:
        row = []
        for enriched_pixel in row_enriched_pixels:
            if enriched_pixel[9] > edge_parameter:
                if enriched_pixel[10] + enriched_pixel[11] + enriched_pixel[12] > threshold:
                    row.append([enriched_pixel[0], 0, 0])
                else:
                    row.append([enriched_pixel[0], enriched_pixel[1], enriched_pixel[2]])
            else:
                # row.append([0, 0, enriched_pixel[2]])
                row.append([0, 0, 0])
        bitmap.append(row)
    return bitmap

def get_bitmap(image):
    """ transform image to bitmap array """
    bitmap = []
    for x in range(image.size[0]):
        row = []
        for y in range(image.size[1]):
            pixels = image.getpixel((x, y))
            # print([pixels[0], pixels[1], pixels[2]])
            row.append([pixels[0], pixels[1], pixels[2]])
        bitmap.append(row)
    return bitmap

def get_preprocessed_bitmap(bitmap):
    """ transform bitmap to enriched bitmap
        R, G, B, x, y, x_relative, y_relative, x_from_center, y_from_center, r_from_center, R_relative, G_relative, B_relative
    """
    enriched_bitmap = []
    size_x = len(bitmap)
    size_y = len(bitmap[0])
    for x, row_pixels in enumerate(bitmap):
        row = []
        for y, pixel in enumerate(row_pixels):
            x_from_center = (x-size_x/2)/(size_x/2)
            y_from_center = (y-size_y/2)/(size_y/2)
            r_from_center = 1-(x_from_center**2 + y_from_center**2)**0.5
            enriched_pixel = [pixel[0], pixel[1], pixel[2], x, y, x/size_x, y/size_y, x_from_center, y_from_center, r_from_center, pixel[0]/255, pixel[1]/255, pixel[2]/255]
            row.append(enriched_pixel)
        enriched_bitmap.append(row)
    return enriched_bitmap

def get_image(bitmap):
    """ transform bitmap to image """
    size_x = len(bitmap)
    size_y = len(bitmap[0])
    image = Image.new(mode="RGB", size=(size_x, size_y))
    for x, row_pixels in enumerate(bitmap):
        for y, pixel in enumerate(row_pixels):
            image.putpixel((x, y), (pixel[0], pixel[1], pixel[2]))
    # image.show()
    return image

def get_initial_points(zone_count, model_dimension):
    """ returns list of simple zones """
    zones = []
    for i in range(zone_count):
        zones.append([i/zone_count]*model_dimension)
    # print(zones)
    return zones

def count_species(bitmap):
    """ it will count all different colors """
    colors = {}

    for row_pixels in bitmap:
        for pixel in row_pixels:
            if str(pixel) in colors.keys():
                colors[str(pixel)] = colors[str(pixel)] + 1
            else:
                colors[str(pixel)] = 1
    return colors

def clear_species(colors, threshold):
    """ eliminate species based on count of species """
    cleared = {}
    for specie in colors.keys():
        if colors[specie] > threshold:
            cleared[specie] = colors[specie]
    return cleared

def eliminate_species_in_bitmap(bitmap, white_list):
    """ eliminate colors based on whitelist """
    cleared_bitmap = []
    for row_pixels in bitmap:
        row = []
        for pixel in row_pixels:
            if str(pixel) in white_list.keys() and str(pixel) != '[0, 0, 0]':
                # row.append([255, 255, 255])
                row.append(pixel)
            else:
                row.append([0, 0, 0])
        cleared_bitmap.append(row)
    return cleared_bitmap

def array_to_bitmap(input_array):
    """ convert array to bitmap """
    input_array = np.transpose(input_array)
    bitmap = []
    for row_pixels in input_array:
        row = []
        for pixel in row_pixels:
            if pixel == 0.0:
                row.append([0, 0, 0])
            else:
                row.append([255, 255, 255])

        bitmap.append(row)
    return bitmap

def separate_to_blobs(bitmap):
    """ separate blobs to separate items """
    pixels_in_use = []
    blobs = []

    used_colors = []

    image = Image.open('output/test_r.tif')

    # Converting the image to RGB mode
    image = image.convert("RGB")

    color = list(np.random.choice(range(256), size=3))
    for x, row_pixels in enumerate(bitmap):
        if x % 2 == 0:
            continue
        for y, pixel in enumerate(row_pixels):
            if y % 2 == 0:
                continue
            if pixel != [0, 0, 0]:
                # if image.getpixel((x, y)) != (color[0], color[1], color[2]):
                if image.getpixel((x, y)) == (0, 255, 0):
                    while color in used_colors:
                        color = list(np.random.choice(range(256), size=3))
                    ImageDraw.floodfill(image, (x, y), (color[0], color[1], color[2]), thresh=1)
                    used_colors.append(color)
                    # print(used_colors)
                    # print(len(used_colors))
                    # image.show()
                    # input()
    image.save('output/test_fill.tif')
    print(used_colors)
    print(len(used_colors))
    return get_bitmap(image)

def separate_to_blobs_direct(bitmap, zone_count):
    """ separate blobs to separate items """
    print(type(bitmap))
    numpy_array = np.asanyarray(bitmap)
    image = imread('output/test_r.tif')
    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1, min_sigma=10)
    points = blobs_log[:, 0:2]

    output = []
    for point in points:
        output.append([point[0], point[1]])
    print(output)
    print(len(blobs_log))

    bitmap_with_points = []
    for x, row_pixels in enumerate(bitmap):
        row = []
        for y, pixel in enumerate(row_pixels):
            if [y, x] in output:
                row.append([255, 0, 0])
            else:
                row.append(pixel)
        bitmap_with_points.append(row)
    return bitmap_with_points

def create_score_map(bitmap, colors, edge_parameter):
    """ transform blobs from bitmap to separate numpy array score maps """
    score_maps = []
    size_x = len(bitmap)
    size_y = len(bitmap[0])

    for color in colors:
        if color != '[255, 0, 0]':
            score_map = np.zeros((size_x, size_y), dtype=np.int16)
            for x, row_pixels in enumerate(bitmap):
                for y, pixel in enumerate(row_pixels):
                    x_from_center = (x-size_x/2)/(size_x/2)
                    y_from_center = (y-size_y/2)/(size_y/2)
                    r_from_center = 1-(x_from_center**2 + y_from_center**2)**0.5
                    if r_from_center > edge_parameter:
                        if str(pixel) == color:
                            score_map[x][y] = 1
                        else:
                            score_map[x][y] = -1000
            score_maps.append(score_map)
            # break
    return score_maps

def get_score(score_map, cx, cy, r):
    """ get score of pixel based on his position and R """
    score = 0
    mask = 0

    # buffer for x & y indices
    indices_x = list()
    indices_y = list()

    # lower and upper index range
    x_lower, x_upper = int(max(cx-r, 0)), int(min(cx+r, score_map.shape[1]-1))
    y_lower, y_upper = int(max(cy-r, 0)), int(min(cy+r, score_map.shape[0]-1))
    range_x = range(x_lower, x_upper)
    range_y = range(y_lower, y_upper)

    # loop over all indices
    for y, x in itertools.product(range_y, range_x):
        # check if point lies within radius r
        if (x-cx)**2 + (y-cy)**2 < r**2:
            indices_y.append(y)
            indices_x.append(x)

    score = np.sum(score_map[(indices_y, indices_x)])
    # print([(indices_y, indices_x)])
    # print(len(score_map[(indices_y, indices_x)]))
    # print(score_map[(indices_y, indices_x)])
    # print(score)
    return score

def evaluate_score_map(score_map):
    """ test each pixel and if its part of color, evaluate it """
    # evaluated_map = np.zeros((score_map.shape[0], score_map.shape[1], 2), dtype=np.int16) # for NN
    evaluation_list = []

    for y, row in enumerate(score_map):
        for x, pixel in enumerate(row):
            if pixel == 1:
                # print(x, y, pixel)
                # evaluated_map[x][y] = find_best_score(score_map, x, y) # for NN
                # print(evaluated_map[x][y])
                evaluation_list.append([np.asarray([x, y]), find_best_score(score_map, x, y)])
    return evaluation_list

def find_best_score(score_map, x, y):
    """ find best r based on score for pixel of score map """
    r = 1
    score = 1
    best_score = 1
    best_r = 1

    # print(r, score) # nice for plots
    while score > 0:
        r += 1
        score = get_score(score_map, x, y, r)
        if score > best_score:
            best_score = score
            best_r = r
        # print(r, score) # nice for plots
    return best_r, best_score

def king_of_the_hill(evaluation_list):
    """ find the king by the process of elimination """
    kings = []
    centers = []
    minimum_r = 3
    soft_parameter = 0.7
    # too_similar = []
    for point_A in evaluation_list:
        alive = True
        # if str(point_A) in too_similar:
            # break
        for point_B in evaluation_list:
            if point_A[0][0] == point_B[0][0] and point_A[0][1] == point_B[0][1] and point_A[1] == point_B[1]:
                continue
            R = np.linalg.norm(point_A[0]-point_B[0])
            r_A = point_A[1][0]
            score_A = point_A[1][1]
            r_B = point_B[1][0]
            score_B = point_B[1][1]
            if r_A < minimum_r:
                alive = False
                break
            if R > r_A:
                continue
            if r_A < r_B:
                alive = False
                break
            elif r_A == r_B:
                if score_A < score_B*soft_parameter:
                    alive = False
                    break
        if alive:
            for king in kings:
                R = np.linalg.norm(point_A[0]-king[0])
                if R < king[1][0]*soft_parameter or R < minimum_r:
                    alive = False
            if alive:
                kings.append(point_A)
                centers.append([point_A[0][0], point_A[0][1]])
    # print(kings)
    # print(len(kings))

    return kings, centers

def evaluate_score_maps(score_maps, bitmap):
    """ evaluate all blobs """
    total_centers = []

    for score_map in score_maps:
        _, centers = king_of_the_hill(evaluate_score_map(score_map))
        total_centers += centers

    print(len(total_centers))
    print(len(total_centers)/167)
    print(abs(len(total_centers)-167)/167)

    enriched_bitmap = []
    for x, row_pixels in enumerate(bitmap):
        row = []
        for y, pixel in enumerate(row_pixels):
            if pixel != [255,0,0] and pixel != [0,0,0]:
                if [y, x] in total_centers:
                    row.append([255, 0, 0])
                else:
                    row.append([255, 255, 255])
            else:
                row.append([0, 0, 0])

        enriched_bitmap.append(row)

    return enriched_bitmap, total_centers

def basic_statistic_about_blobs(bitmap, total_centers, colors):
    """ this function will find out average size of surface per colony and also supplement not counted blobs at least as one colony """
    blobs_statistic = {}
    for color in colors.keys():
        area = colors[color]
        centers = 0
        blobs_statistic[color] = [centers, area]

    print(total_centers)

    for center in total_centers:
        print(center)
        print(bitmap[center[0], center[1]])

    print(blobs_statistic)

def create_nn_model(X_train, X_test, X_validation, Y_train, Y_test, Y_validation):
    """ train and test nn model """
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(3,1)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(7, activation='softmax')
            ])

    loss_function = tf.losses.CategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=3)

    model.evaluate(X_test, Y_test, verbose=2)

    model.save('model.h5')

    return model

def flat_bitmap(enriched_bitmap):
    """ make bitmap flat """
    pixels = []
    edges = []
    x = len(enriched_bitmap)
    y = len(enriched_bitmap[0])
    for row in enriched_bitmap:
        for pixel in row:
            pixels.append([pixel[10], pixel[11], pixel[12]])
            edges.append(pixel[9])
    return pixels, x, y, edges

def put_it_in_the_box(pixels, x, y):
    """ make it in the box """
    bitmap = []
    index = 0
    for _ in range(x):
        row = []
        for _ in range(y):
            row.append(pixels[index])
            index += 1
        bitmap.append(row)
    return bitmap

def use_nn_model(enriched_bitmap, edge_parameter, model):
    """ uses nn model on bitmap """
    bitmap = []
    nice_bitmap = []

    pixels, x, y, edges = flat_bitmap(enriched_bitmap)
    results = model.predict(pixels)
    results = put_it_in_the_box(results, x, y)

    index = 0
    for x, result_row in enumerate(results):
        row = []
        nice_row = []
        for y, result in enumerate(result_row):
            if edges[index] > edge_parameter:
                result_index = np.argmax(result)
                if result_index == 0:
                    nice_row.append([enriched_bitmap[x][y][0], 0, 0])
                    row.append([255, 0, 0])
                elif result_index == 1:
                    nice_row.append([0, enriched_bitmap[x][y][1], 0])
                    row.append([0, 255, 0])
                elif result_index == 2:
                    nice_row.append([0, 0, enriched_bitmap[x][y][2]])
                    row.append([0, 0, 255])
                elif result_index == 3:
                    nice_row.append([enriched_bitmap[x][y][2], 0, enriched_bitmap[x][y][2]])
                    row.append([255, 0, 255])
                elif result_index == 4:
                    nice_row.append([0, enriched_bitmap[x][y][1], enriched_bitmap[x][y][2]])
                    row.append([0, 255, 255])
                elif result_index == 5:
                    nice_row.append([enriched_bitmap[x][y][0], enriched_bitmap[x][y][1], 0])
                    row.append([255, 255, 0])
            else:
                nice_row.append([0, 0, 0])
                row.append([0, 0, 0])
            index += 1
        bitmap.append(row)
        nice_bitmap.append(nice_row)
    return bitmap, nice_bitmap

def get_low_high(points):
    """ this will get threshold """
    R_min, R_max, G_min, G_max, B_min, B_max = 0, 0, 0, 0, 0, 0
    for point in points:
        if R_min > point[0]:
            R_min = point[0]
        if R_max < point[0]:
            R_max = point[0]
        if G_min > point[1]:
            G_min = point[1]
        if G_max < point[1]:
            G_max = point[1]
        if B_min > point[2]:
            B_min = point[2]
        if B_max < point[2]:
            B_max = point[2]
    low = np.array([R_min, G_min, B_min])
    high = np.array([R_max, G_max, B_max])
    return low, high

def separate_agar(image, edge_parameter):
    """"""
    # import cv2
    # # sample_image = cv2.imread(image)

    # out_img = cv2.imread('output/reference.tif').astype(np.int8)
    # tar_img = cv2.imread(image).astype(np.int8)

    # colors = []
    # for color in out_img:
    #     if color not in colors:
    #         colors.append([color[0], color[1], color[2]])

    # tar_img = cv2.resize(out_img, (tar_img.shape[0], tar_img.shape[1]))

    # result = out_img - tar_img


    # img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    # points = [[46,36,9], [50,43,15], [53,41,17], [50,37,18], [52,41,13], [48,41,15], [50,43,25], [47,44,37]]
    # low, high = get_low_high(points)

    # mask = cv2.inRange(img, low, high)
    # mask = cv2.bitwise_not(mask)

    # result = cv2.bitwise_and(img, img, mask=mask)
    # result = cv2.fastNlMeansDenoisingColored(result,None,10,10,7,21)


    # bitmap = []
    # for row in tar_img:
    #     bitmap_row = []
    #     for pixel in row:
    #         if np.any(np.all(pixel == colors, axis=1)):
    #             bitmap_row.append(pixel)
    #         else:
    #             bitmap_row.append([0,0,0])
    #     bitmap.append(bitmap_row)
    # image = get_image(bitmap)

    # # image.show()
    # image.save('output/tet.tif')

    # image = get_image(get_bitmap_just_center(image, edge_parameter))
    # image = rgb2gray(image)

    # blurred_image = filters.gaussian(image, sigma=1.0)
    # t = filters.threshold_otsu(blurred_image)
    # mask = blurred_image < 0.1

    # print(image.shape)

    # mask = image[:, :] < t
    # image[mask] = 0

    # bitmap = []
    # for row in image:
    #     bitmap_row = []
    #     for pixel in row:
    #         bitmap_row.append([int(pixel*255), int(pixel*255), int(pixel*255)])
    #     bitmap.append(bitmap_row)
    # image = get_image(bitmap)

    # image.show()

    segments = slic(image, n_segments=4, convert2lab=True)
    print(segments, np.unique(segments))

    used_colors = []
    for _ in range(len(np.unique(segments)) + 5):
        color = [0,0,0]
        while color in used_colors:
            color = list(np.random.choice(range(256), size=3))
        used_colors.append(color)

    # print(len(used_colors))
    bitmap = []
    for row in segments:
        bitmap_row = []
        for pixel in row:
            bitmap_row.append(used_colors[pixel])
        bitmap.append(bitmap_row)
    image = get_image(bitmap)

    # image.show()
    image.save('output/tet.tif')

    exit(0)

def get_bitmap_just_center(image, edge_parameter):
    """ transform image to bitmap array with separated misc edge """
    bitmap = []
    size_x = image.size[0]
    size_y = image.size[1]

    for x in range(size_x):
        row = []
        for y in range(size_y):
            x_from_center = (x-size_x/2)/(size_x/2)
            y_from_center = (y-size_y/2)/(size_y/2)
            r_from_center = 1-(x_from_center**2 + y_from_center**2)**0.5
            if r_from_center > edge_parameter:
                pixels = image.getpixel((x, y))
                row.append([pixels[0], pixels[1], pixels[2]])
            else:
                row.append([0,0,0])
        bitmap.append(row)
    return bitmap

def main(args):
    """
    	The main function takes in a args parameter, which should be an object containing various parameters needed for the function to execute properly. The function iterates through all files in a directory with a certain file extension, applies Gaussian blur to the image, generates a bitmap, processes the bitmap using the processed_bitmap_by_KM function, and saves the resulting images with a modified filename. The function also prints out the name of the processed file and various scores associated with the processed bitmap.

	:param args: an object containing various parameters needed for the function to execute properly, including edge_parameter, zone_count, model_dimension, gauss_blur_parametr, and sample_file_name.
	:return: None.
    """
    print('name', 'R', 'G', 'B', 'V', 'C', 'Y', 'N')

    edge_parameter = args.edge_parameter
    zone_count = args.zone_count
    model_dimension = args.model_dimension
    gauss_blur_parametr = args.gauss_blur_parametr
    sample_file_name = args.sample_file_name

    for file_name in glob.glob(f'{args.dir}*{args.e}'):
        # print('\t', file_name)
        with Image.open(file_name) as image:
            width, height = image.size
            image = image.resize((width // 4, height // 4))
            width, height = image.size
            # separate_agar(image, edge_parameter)

            # print(width, height)
            image = image.filter(ImageFilter.GaussianBlur(gauss_blur_parametr))
            bitmap = get_bitmap(image=image)
            enriched_bitmap = get_preprocessed_bitmap(bitmap)
            bitmap, nice_bitmap, model, score = processed_bitmap_by_KM(enriched_bitmap, edge_parameter=edge_parameter, zone_count=zone_count, initial_points_relative_colors=get_initial_points(zone_count, model_dimension), model_dimension=model_dimension)
            image = get_image(bitmap).save(file_name.replace('.tif', '_r.tif'))
            get_image(nice_bitmap).save(file_name.replace('.tif', '_rn.tif'))
            print(file_name.split('/')[-1].split('.')[0], score[0], score[1], score[2], score[3], score[4], score[5], score[6])

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
