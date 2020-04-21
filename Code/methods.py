# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:23:31 2020

@author: Alex
"""

import numpy as np
import contextlib as ctxlib
import collections
import sklearn
import random
import os

from matplotlib.image import imread
from math import inf, nan


def reshapeImgs(images):
    newImages = []
    for image in images:
        # Remove rgb dimension
        img = image[:, :, 0]
        # 200, 300 to 60.000 pixels
        img = np.ravel(img)
        newImages.append(img)
    
    return newImages


def loadImages(path):
    # return array of images

    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        img = imread(os.path.join(path, image))
        #img = imread(path + image)
        loadedImages.append(img)

    return loadedImages


def generateYArray(rock_imgs, paper_imgs, scissor_imgs):
    imgs = rock_imgs + paper_imgs + scissor_imgs
    
    # Rock
    all_imgs = np.zeros(len(imgs))
    # paper
    all_imgs[len(rock_imgs):len(rock_imgs) + len(paper_imgs)] = 1
    # scissors
    all_imgs[len(rock_imgs) + len(paper_imgs):] = 2
    
    all_imgs = list(all_imgs)
    
    return all_imgs