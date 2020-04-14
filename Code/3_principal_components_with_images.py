# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:30:12 2020

@author: Alex
"""

from os import listdir
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import scale
#%%
# Data preperation and getting data from files
# ref:
# https://stackoverflow.com/questions/36774431/how-to-load-images-from-a-directory-on-the-computer-in-python?fbclid=IwAR2KJY6YPoo20MI4VUsB6oajFXiRf0twNR73BBIYGsvG1mpGUZQgwo0pf_Q

# Paths
#path_working_dir = "C:/Users/flole/Desktop/dev/MAL/itmal/Own_Prj/rps-cv-images/"
path_working_dir = os.path.dirname(os.path.realpath(__file__))
path_paper = path_working_dir + "\\..\\rockpaperscissors\\paper\\"
path_rock = path_working_dir + "\\..\\rockpaperscissors\\rock\\"
path_scissors = path_working_dir + "\\..\\rockpaperscissors\\scissors\\"

# Original dataset for comparison
path_paper_pristine = path_working_dir + "\\..\\rockpaperscissors\\rps-cv-images\\paper\\"
path_rock_pristine = path_working_dir + "\\..\\rockpaperscissors\\rps-cv-images\\rock\\"
path_scissors_pristine = path_working_dir + "\\..\\rockpaperscissors\\rps-cv-images\\scissors\\"

def loadImages(path):
    # return array of images
    
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = imread(path + image)
        loadedImages.append(img)

    return loadedImages

rock_imgs = loadImages(path_rock)
paper_imgs = loadImages(path_paper)
scissors_imgs = loadImages(path_scissors)

rock_imgs_pristine = loadImages(path_rock_pristine)
paper_imgs_pristine = loadImages(path_paper_pristine)
scissors_imgs_pristine = loadImages(path_scissors_pristine)

imgs = rock_imgs + paper_imgs + scissors_imgs
imgs_pristine = rock_imgs_pristine + paper_imgs_pristine + scissors_imgs_pristine

# reshape images to not include the rgb parameter (3)
def reshapeImgs(images):
    newImages = []
    for image in images:
        # Remove rgb dimension
        img = image[:, :, 0]
        # 200, 300 to 60.000 pixels
        img = np.ravel(img)
        newImages.append(img)
    
    return newImages

# Are these lines actually used?
reshaped_imgs = reshapeImgs(imgs)
reshaped_imgs_pristine = reshapeImgs(imgs_pristine)

# Testing and showing images
# plt.imshow(imgs[1438])

#%%
# Seeing if we can get some scatter plots with patterns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import scale

pca_3 = PCA(n_components=3)
rock_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(rock_imgs)))
paper_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(paper_imgs)))
scissors_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(scissors_imgs)))

rock_imgs_pca3_pristine = pca_3.fit_transform(scale(reshapeImgs(rock_imgs_pristine)))
paper_imgs_pca3_pristine = pca_3.fit_transform(scale(reshapeImgs(paper_imgs_pristine)))
scissors_imgs_pca3_pristine = pca_3.fit_transform(scale(reshapeImgs(scissors_imgs_pristine)))

#%%

# cleaned up 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
cx.view_init(0, 0)
# Pristine 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3_pristine[:,0], rock_imgs_pca3_pristine[:,1], rock_imgs_pca3_pristine[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3_pristine[:,0], paper_imgs_pca3_pristine[:,1], paper_imgs_pca3_pristine[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3_pristine[:,0], scissors_imgs_pca3_pristine[:,1], scissors_imgs_pca3_pristine[:,2], marker='x', color='orange')
cx.view_init(0, 0)


# cleaned up 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
cx.view_init(0, 180)
# Pristine 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3_pristine[:,0], rock_imgs_pca3_pristine[:,1], rock_imgs_pca3_pristine[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3_pristine[:,0], paper_imgs_pca3_pristine[:,1], paper_imgs_pca3_pristine[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3_pristine[:,0], scissors_imgs_pca3_pristine[:,1], scissors_imgs_pca3_pristine[:,2], marker='x', color='orange')
cx.view_init(0, 180)


# cleaned up 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
cx.view_init(0, 60)
# Pristine 3d plot
fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3_pristine[:,0], rock_imgs_pca3_pristine[:,1], rock_imgs_pca3_pristine[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3_pristine[:,0], paper_imgs_pca3_pristine[:,1], paper_imgs_pca3_pristine[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3_pristine[:,0], scissors_imgs_pca3_pristine[:,1], scissors_imgs_pca3_pristine[:,2], marker='x', color='orange')
cx.view_init(0, 60)