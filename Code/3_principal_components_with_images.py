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
path_paper = path_working_dir + "\\paper\\"
path_rock = path_working_dir + "\\rock\\"
path_scissors = path_working_dir + "\\scissors\\"


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

imgs = rock_imgs + paper_imgs + scissors_imgs

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

reshaped_imgs = reshapeImgs(imgs)

# Testing and showing images
plt.imshow(imgs[1438])

#%%
# Seeing if we can get some scatter plots with patterns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import scale

pca_3 = PCA(n_components=3)
rock_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(rock_imgs)))
paper_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(paper_imgs)))
scissors_imgs_pca3 = pca_3.fit_transform(scale(reshapeImgs(scissors_imgs)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
ax.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
ax.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
ax.view_init(180, 130)

fig = plt.figure()
bx = fig.add_subplot(111, projection='3d')
bx.set_xlabel('X')
bx.set_ylabel('Y')
bx.set_zlabel('Z')
bx.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
bx.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
bx.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
bx.view_init(0, 130)

fig = plt.figure()
cx = fig.add_subplot(111, projection='3d')
cx.set_xlabel('X')
cx.set_ylabel('Y')
cx.set_zlabel('Z')
cx.scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], marker='o', color='blue')
cx.scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2], marker='^', color='green')
cx.scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1], scissors_imgs_pca3[:,2], marker='x', color='orange')
cx.view_init(-110, 90)


#%%
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(rock_imgs_pca3[:,0], rock_imgs_pca3[:,1], rock_imgs_pca3[:,2], projection='3d')
axs[0, 0].set_title('Rock Pattern')
axs[0, 1].scatter(paper_imgs_pca3[:,0], paper_imgs_pca3[:,1], paper_imgs_pca3[:,2])
axs[0, 1].set_title('Paper Pattern')
axs[1, 0].scatter(scissors_imgs_pca3[:,0], scissors_imgs_pca3[:,1],scissors_imgs_pca3[:,2])
axs[1, 0].set_title('Scissor Pattern')

#%%

X_digits_pca2_1 = []
X_digits_pca2_3 = []
X_digits_pca2_5 = []
X_digits_pca2_9 = []

index = 0
for digit in y_digits[:]:
    if digit == 1:
        X_digits_pca2_1.append(X_digits_pca2[index])
    elif digit == 3:
        X_digits_pca2_3.append(X_digits_pca2[index])
    elif digit == 5:
        X_digits_pca2_5.append(X_digits_pca2[index])
    elif digit == 9:
        X_digits_pca2_9.append(X_digits_pca2[index])
        
    index = index + 1
  
X_digits_pca2_1 = np.array(X_digits_pca2_1[:])
X_digits_pca2_3 = np.array(X_digits_pca2_3[:])
X_digits_pca2_5 = np.array(X_digits_pca2_5[:])
X_digits_pca2_9 = np.array(X_digits_pca2_9[:])
