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

## Switch with the classifiers suggested in cheat sheat

sgd_model = SGDClassifier(random_state=42)
kneighbours_model = KNeighborsClassifier()

long_sgd_tuning_parameters = {
    'alpha':[0.1, 0.01, 0.001, 0.0001, 0.00001], 
    'penalty': ('l2', 'l1', 'elasticnet'),
    'max_iter':[10, 20 ,30, 50, 70, 100, 500, 1000],
    'l1_ratio': [0.10, 0.15, 0.20, 0.30, 0.5, 0.85],
    'learning_rate':('constant', 'optimal', 'invscaling', 'adaptive'),
    'eta0': [0.1, 0.2, 0.3]
}

kn_tuning_parameters = {
    'algorithm': ('auto', 'ball_tree', 'kd_tree'),
    'leaf_size': [10, 30, 50],
    'n_neighbors': [2, 3, 5],
    'weights': ('uniform', 'distance')
    
}

sgd_tuning_parameters = {
    'alpha':[0.01, 0.001, 0.0001], 
    'penalty': ('l2', 'l1', 'elasticnet'),
    'max_iter':[10, 100, 1000],
    'l1_ratio': [0.15,0.5,0.85]
}

sgd_CV=5
VERBOSE=0

start = time()
sgd_random_tuned = RandomizedSearchCV(
    sgd_model, 
    sgd_tuning_parameters, 
    n_iter=1, 
    random_state=42, 
    cv=sgd_CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1, 
    iid=True
)
#sgd_random_tuned.fit(X_train, y_train)
#sgd_t = time()-start

kn_random_tuned = RandomizedSearchCV(
    kneighbours_model, 
    kn_tuning_parameters, 
    n_iter=1, 
    random_state=42, 
    cv=sgd_CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1, 
    iid=True
)
kn_random_tuned.fit(X_train, y_train)
kn_t = time()-start

# Report result
# sgd_b0, sgd_m0 = FullReport(sgd_random_tuned , X_test, y_test, t)

kn_b0, kn_m0 = FullReport(kn_random_tuned , X_test, y_test, t)
print(kn_b0)
#print(sgd_b0)