# extract main colors by kmeans from sklearn

import math
import os
import sys
from operator import attrgetter, itemgetter

import cv2
import numpy as np
from color_palette_completion.utils.color_convertor import lab_to_rgb, rgb_to_lab
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def color_cluster_kmeans(image, color_numbers):
    # cluster the pixel intensities
    cluster = KMeans(n_clusters=color_numbers)
    cluster.fit(image)

    return cluster


# find the closest pixel in image with cluster centers
def real_centroids(cluster_centers, image):
    real_centers = []
    for c in cluster_centers:
        minDis = math.inf
        tmpC = []
        for i in image:
            tmpDis = math.sqrt(math.pow((c[0] - i[0]), 2) + math.pow((c[1] - i[1]), 2) + math.pow((c[2] - i[2]), 2))
            if minDis > tmpDis:
                minDis = tmpDis
                tmpC = i
        real_centers.append(tmpC)

    return real_centers


def get_colors(image, color_numbers):
    cluster = color_cluster_kmeans(image, color_numbers)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(cluster.labels_)

    # find the closest pixel in image with cluster centers (result seems same)
    real_centers = real_centroids(cluster.cluster_centers_, image)
    (colors, colors_lab, rates, palette) = create_palette(hist, real_centers)

    return (colors, colors_lab, rates, palette)


# get historgram of extracted color centroids
def centroid_histogram(labels):
    # grab the number of different clusters and create a histogram
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


# create palette with color rates
def create_palette(hist, centroids):
    minRate = 0.01
    # initialize the palette representing the relative frequency of each color
    # default with RGB color space
    colors = []
    colors_lab = []
    rates = []
    palette = np.zeros((50, 300, 3), dtype="uint8")
    # sort colors decreasing
    hist_centroids = list(zip(hist, centroids))
    hist_centroids.sort(key=itemgetter(0), reverse=True)
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (rate, color) in hist_centroids:
        if rate > minRate:
            # plot the relative percentage of each cluster
            endX = startX + (rate * 300)
            lab = color.astype("uint8").tolist()
            rgb = lab_to_rgb(lab)

            cv2.rectangle(palette, (int(startX), 0), (int(endX), 50), rgb, -1)
            colors_lab.append(lab)
            colors.append(rgb)
            rates.append(rate)
            startX = endX
    # draw border line for easy check in light gray #D3D3D3
    cv2.rectangle(palette, (0, 0), (300, 50), (211, 211, 211), 2)

    return (colors, colors_lab, rates, palette)