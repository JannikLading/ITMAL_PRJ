# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:04:48 2020

@author: flole
"""

# This code and file is heavily inspired by
# https://www.kaggle.com/drgfreeman/principal-component-analysis-visualization?fbclid=IwAR02OvRRJqSRjIGupQqZ7nYrJ3yv5VEdkWT4pgPF5sGTjcEEwJBW5PEozOs
import os
#from PIL import Image as PImage
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

#%% 
# Data preperation and getting data from files
# ref:
# https://stackoverflow.com/questions/36774431/how-to-load-images-from-a-directory-on-the-computer-in-python?fbclid=IwAR2KJY6YPoo20MI4VUsB6oajFXiRf0twNR73BBIYGsvG1mpGUZQgwo0pf_Q

# Paths
# Change if the project is another folder
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
path_working_dir = desktop_path + "/dev/ITMAL_PRJ/rockpaperscissors/"
#path_working_dir = "C:/Users/flole/Desktop/dev/ITMAL_PRJ/rockpaperscissors\"
path_paper = path_working_dir + "paper/"
path_rock = path_working_dir + "rock/"
path_scissors = path_working_dir + "scissors/"


def loadImages(path):
    # return array of images

    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        img = imread(os.path.join(path, image))
        #img = imread(path + image)
        loadedImages.append(img)

    return loadedImages

rock_imgs = loadImages(path_rock)
paper_imgs = loadImages(path_paper)
scissors_imgs = loadImages(path_scissors)

imgs = rock_imgs + paper_imgs + scissors_imgs

#%%
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

#%% 
# y array to define the different options
# rock = 0, paper = 1, scissors = 2
def generateYArray():
    all_imgs = np.zeros(len(imgs))
    all_imgs[len(rock_imgs):len(paper_imgs)] = 1
    all_imgs[len(paper_imgs):] = 2
    
    return all_imgs

y = generateYArray()
#%%
# Testing and showing images
plt.imshow(imgs[1438])

#%%
from sklearn.decomposition import PCA

#%%
def showPCAExplainedVar():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(pca.n_components_), pca.explained_variance_ratio_)
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title('Explained variance ratio by principal component')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title('Cummulative explained variance ratio')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
#%%
pca = PCA(n_components=500)
reshaped_imgs_pca = pca.fit(reshaped_imgs)

#%%
showPCAExplainedVar()

#%%
pca = PCA(n_components=200)
reshaped_imgs_pca = pca.fit(reshaped_imgs)

#%%
showPCAExplainedVar()

#%%
pca.components_.shape

#%%
imgs_pca = pca.transform(reshaped_imgs)
imgs_pca.shape

#%%
# Inverse pcs on compressed images
inv_imgs_from_pca = pca.inverse_transform(imgs_pca)


#%%
# Plot differences between original and pca image
def show_img_pcs(index):
    plt.figure(figsize=(16, 4))
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(reshaped_imgs[index].reshape(200, 300), cmap="gray");
    plt.title('Original image')
    plt.xticks(())
    plt.yticks(())
    
    #Display principal components magnitude
    plt.subplot(1, 3, 2)
    img_pc = pca.transform([reshaped_imgs[index]])
    plt.bar(range(1, img_pc.shape[1] + 1), img_pc[0,:])
    plt.title('Image principal components magnitude')
    plt.xlabel('Principal component')
    plt.ylabel('Magnitude')
    
    # Display reconstituted image
    plt.subplot(1, 3, 3)
    plt.imshow(pca.inverse_transform(img_pc).reshape(200, 300), cmap="gray")
    plt.title('Image reconstituted from principal components')
    plt.xticks(())
    plt.yticks(())
    
    plt.tight_layout()
    plt.show()
    
#%%
# Rock
show_img_pcs(725)
# Paper
show_img_pcs(726)
#Scissors
show_img_pcs(2000)