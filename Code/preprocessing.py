# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:49:27 2020

@author: Jannik
"""

from os import listdir
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
path_working_dir = os.path.dirname(os.path.realpath("__file__"))
path_paper = path_working_dir + "\\..\\rockpaperscissors\\paper\\"
path_rock = path_working_dir + "\\..\\rockpaperscissors\\rock\\2O9XPBJRT119drWX.png"
path_scissors = path_working_dir + "\\..\\rockpaperscissors\\scissors\\"

#%%

img = imread(path_rock)
                     
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



gray = rgb2gray(img)    
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

#%%
import cv2

def removeGreenScreen(image_path):
    img = cv2.imread(image_path)
    
    image_copy = np.copy(img)

    lower_green = np.array([0, 100, 0])     ##[R value, G value, B value]
    upper_green = np.array([100, 255, 100])

    mask = cv2.inRange(image_copy, lower_green, upper_green)

    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    return masked_image


img = removeGreenScreen(path_rock)
plt.imshow(img)
plt.show()

