# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:28:47 2020

@author: Alex
"""
import sys, os
# from os import listdir
import inspect
import pathlib

from methods import loadImages, reshapeImgs, generateYArray

# Setting img source path
## __file__ DOES NOT WORK IF YOU RUN CELLS!! 
## Have to run whole file at least once for the variable!!!
path_working_dir = os.path.dirname(os.path.abspath(__file__))
path_paper = path_working_dir + "..\\..\\rockpaperscissors\\paper\\"
path_rock = path_working_dir + "..\\..\\rockpaperscissors\\rock\\"
path_scissors = path_working_dir + "..\\..\\rockpaperscissors\\scissors\\"

#%%  

# Loading imgs
rock_imgs = loadImages(path_rock)
paper_imgs = loadImages(path_paper)
scissors_imgs = loadImages(path_scissors)

imgs = rock_imgs + paper_imgs + scissors_imgs
y = generateYArray(rock_imgs, paper_imgs, scissors_imgs)
X = imgs

test_train_split_80 = int(len(imgs) * 0.8)
X_train, X_test = X[:test_train_split_80], X[test_train_split_80:]
y_train, y_test = y[:test_train_split_80], y[test_train_split_80:]

#%%
# Running The LinearSVC model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import LinearSVC
from time import time
#clf = LinearSVC(random_state=0, max_iter=10000)
#clf.fit(X_imgs_training, y_training)
#y_pred = clf.predict(X_imgs_test)

linearsvc_model = LinearSVC()
linearsvc_tuning_parameters = {
        "penalty": ('l1','l2'),#"penalty": ("l1", "l2"),
        "loss": ("hinge", "squared_hinge"),#"loss": ("hinge"),#
        "dual": [False, True],
        "tol": [1e-3, 1e-2, 1e-4],
        "C": [0.2, 0.5, 1, 1.5, 2],
        "multi_class": ("ovr", "crammer_singer"),
        "fit_intercept": [False, True],
        "intercept_scaling": [0.2, 0.5, 1, 1.5, 2],
        "max_iter": [500, 1000, 1500, 2000, 5000, 8000, 10000]
        }

linearsvc_random_tuned = RandomizedSearchCV(
        linearsvc_model,
        linearsvc_tuning_parameters,
        n_iter = 500,
        random_state = 42,
        cv=5,
        scoring="f1_micro",
        verbose=True,
        n_jobs=-1,
        iid=True,
        error_score=0.0 # needed to not get error from bad combinations
        )

linearsvc_grid_tuned = GridSearchCV(
        linearsvc_model,
        linearsvc_tuning_parameters,
        cv=5,
        scoring="f1_micro",
        verbose=True,
        n_jobs=-1,
        iid=True,
        error_score=0.0)


start = time()

#linearsvc_grid_tuned.fit(X_imgs_training, y_training)

# linearsvc_random_tuned.fit(X_imgs_training, y_training)

linearsvc_t = time()-start

#linearsvc_b0, linearsvc_m0 = FullReport(linearsvc_random_tuned, X_imgs_test, y_test, linearsvc_t)

#linearsvc_b0, linearsvc_m0 = FullReport(linearsvc_grid_tuned, X_imgs_test, y_test, linearsvc_t)

