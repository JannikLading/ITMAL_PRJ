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

#%% Jannik path variables
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')))
path_working_dir = desktop_path + "\\ITMAL_PRJ\\rockpaperscissors\\"
path_paper = path_working_dir + "paper\\"
path_rock = path_working_dir + "rock\\"
path_scissors = path_working_dir + "scissors\\"

#%% Florent other pc
path_working_dir = "E:" + "/dev/ITMAL_PRJ/rockpaperscissors/"
#path_working_dir = "C:/Users/flole/Desktop/dev/ITMAL_PRJ/rockpaperscissors\"
path_paper = path_working_dir + "paper/"
path_rock = path_working_dir + "rock/"
path_scissors = path_working_dir + "scissors/"
#%% General purpose, should work on all
path_working_dir = os.path.dirname(os.path.realpath(__file__))
path_paper = path_working_dir + "\\..\\rockpaperscissors\\paper\\"
path_rock = path_working_dir + "\\..\\rockpaperscissors\\rock\\"
path_scissors = path_working_dir + "\\..\\rockpaperscissors\\scissors\\"


#%%

def loadImages(path):
    # return array of images

    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        #img = imread(os.path.join(path, image))
        img = imread(path + image)
        loadedImages.append(img)
        
    return loadedImages

rock_imgs = loadImages(path_rock)
paper_imgs = loadImages(path_paper)
scissors_imgs = loadImages(path_scissors)

imgs = rock_imgs + paper_imgs + scissors_imgs

#%%
# Testing and showing images
plt.imshow(imgs[1438])

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
    # Rock
    all_imgs = np.zeros(len(imgs))
    # paper
    all_imgs[len(rock_imgs):len(rock_imgs) + len(paper_imgs)] = 1
    # scissors
    all_imgs[len(rock_imgs) + len(paper_imgs):] = 2
    
    all_imgs = list(all_imgs)
    
    return all_imgs

y = generateYArray()

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
print(imgs_pca.shape)

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

#%%
# Shuffling the data
import random
random.seed(42)

# Copy the datasets, otherwise the actual datasets are going to be shuffled
shuffled_imgs = imgs_pca.copy()
shuffled_y = y.copy()

# Zip the datasets to be able to shuffle them in the same order
c = list(zip(shuffled_imgs, shuffled_y))

#%%
# Shuffle the datasets, so that they match in order
random.shuffle(c)

#%%
# Get the shuffled lists
shuffled_imgs, shuffled_y = zip(*c)
shuffled_imgs = list(shuffled_imgs)
shuffled_y = list(shuffled_y) 

#%%
# Test if it works. Not the same shape anymore, the data is compressed
index = 555
#plt.imshow(shuffled_imgs[index].reshape(200, 300), cmap="gray");
#print(shuffled_y[index])

#%%
# Split data into training- and test sets

# Training
X_imgs_training = shuffled_imgs[500:]
y_training = shuffled_y[500:]
# Test
X_imgs_test = shuffled_imgs[:500]
y_test = shuffled_y[:500]
#%%
#plt.imshow(X_imgs_test[index].reshape(200, 300), cmap="gray");
#print(y_test[index])

#%%
# Quick test of the 3 algorithms
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


linearSVC = LinearSVC(random_state=0, max_iter=10000)
linearSVC.fit(X_imgs_training, y_training)
linsvc_y_pred = linearSVC.predict(X_imgs_test)
print(f"LinearSVC F1 Score: {f1_score(y_test, linsvc_y_pred, average='micro')}")

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_imgs_training, y_training)
neigh_y_pred = neigh.predict(X_imgs_test)
print(f"KNeighbors F1 Score: {f1_score(y_test, neigh_y_pred , average='micro')}")

svc = SVC(gamma='auto')
svc.fit(X_imgs_training, y_training)
svc_y_pred = svc.predict(X_imgs_test)
print(f"SVC F1 Score: {f1_score(y_test, svc_y_pred, average='micro')}")

#%% Best model selection and hyperparameter tuning
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score

#%% LinearSearchSCV model tuning parameters 
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

#%% LinearSearchSCV search
linearsvc_model = LinearSVC()
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

#%% LinearSearchSCV model run and fit
start_grid = time()

linearsvc_grid_tuned.fit(X_imgs_training, y_training)

linearsvc_t_1 = time()-start_grid


start_rand = time()

linearsvc_random_tuned.fit(X_imgs_training, y_training)

linearsvc_t_rand = time()-start_rand

#%% LinearSearchSCV model report
linearsvc_rand_b0, linearsvc_rand_m0 = FullReport(linearsvc_random_tuned, X_imgs_test, y_test, linearsvc_t_rand)

linearsvc_grid_b0, linearsvc_grid_m0 = FullReport(linearsvc_grid_tuned, X_imgs_test, y_test, linearsvc_t_grid)

#%% KNeighbours model tuning parameters
kn_tuning_parameters = {
    'algorithm': ('auto', 'ball_tree', 'kd_tree'),
    'leaf_size': [10, 30, 50],
    'n_neighbors': [2, 3, 5],
    'weights': ('uniform', 'distance'),
    'p': [1, 2]
}

CV=5
VERBOSE=0

#%% KNeighbours model Randomized search CV
start = time()
random_neigh_tuned = RandomizedSearchCV(
    neigh, 
    kn_tuning_parameters, 
    n_iter=20, 
    random_state=42, 
    cv=CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1, 
    iid=True
)
random_neigh_tuned.fit(X_imgs_training, y_training)
y_pred = random_neigh_tuned.predict(X_imgs_test)
t = time()-start

# Report result
b0, m0= FullReport(random_neigh_tuned , X_imgs_test, y_test, t)

#%% KNeighbours model Grid search CV
# Run GridSearchCV for the model
start = time()
grid_tuned = GridSearchCV(neigh, kn_tuning_parameters, cv=CV, scoring='f1_micro', verbose=VERBOSE, n_jobs=-1, iid=True)
grid_tuned.fit(X_imgs_training, y_training)
t = time()-start

# Report result
b0, m0= FullReport(grid_tuned , X_imgs_test, y_test, t)

#%% SCV model tuning parameters 
svc_tuning_parameters = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree' : [1, 3, 5, 10],
    'gamma' : [1e-3, 1e-4, 1e-5],
    'C' : [1, 10, 100, 1000, 1000],
    'max_iter' : [10, 100, 1000, 10000],
}
CV=5
VERBOSE=0

#%% SCV model Randomized search CV
start = time()
svc_tuned = RandomizedSearchCV(
    SVC(), 
    svc_tuning_parameters, 
    n_iter=1000, 
    random_state=42, 
    cv=CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1, 
    iid=True
)
svc_tuned.fit(X_imgs_training, y_training)
y_pred = svc_tuned.predict(X_imgs_test)
t = time()-start

# Report result
b0, m0= FullReport(svc_tuned , X_imgs_test, y_test, t)

#%% SVC model Grid search CV
# Run GridSearchCV for the model
start = time()
svc_grid_tuned = GridSearchCV(SVC(), svc_tuning_parameters, cv=CV, scoring='f1_micro', verbose=VERBOSE, n_jobs=-1, iid=True)
svc_grid_tuned.fit(X_imgs_training, y_training)
t = time()-start

# Report result
b0, m0= FullReport(svc_grid_tuned , X_imgs_test, y_test, t)
#%% Report functions (reference: MAL Lesson 9 gridsearch assignment)
currmode="N/A" # GLOBAL var!

def SearchReport(model): 
    
    def GetBestModelCTOR(model, best_params):
        def GetParams(best_params):
            ret_str=""          
            for key in sorted(best_params):
                value = best_params[key]
                temp_str = "'" if str(type(value))=="<class 'str'>" else ""
                if len(ret_str)>0:
                    ret_str += ','
                ret_str += f'{key}={temp_str}{value}{temp_str}'  
            return ret_str          
        try:
            param_str = GetParams(best_params)
            return type(model).__name__ + '(' + param_str + ')' 
        except:
            return "N/A(1)"
        
    print("\nBest model set found on train set:")
    print()
    print(f"\tbest parameters={model.best_params_}")
    print(f"\tbest '{model.scoring}' score={model.best_score_}")
    print(f"\tbest index={model.best_index_}")
    print()
    print(f"Best estimator CTOR:")
    print(f"\t{model.best_estimator_}")
    print()
    try:
        print(f"Grid scores ('{model.scoring}') on development set:")
        means = model.cv_results_['mean_test_score']
        stds  = model.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("\t[%2d]: %0.3f (+/-%0.03f) for %r" % (i, mean, std * 2, params))
            i += 1
    except:
        print("WARNING: the random search do not provide means/stds")
    
    global currmode                
    assert "f1_micro"==str(model.scoring), f"come on, we need to fix the scoring to be able to compare model-fits! Your scoreing={str(model.scoring)}...remember to add scoring='f1_micro' to the search"   
    return f"best: dat={currmode}, score={model.best_score_:0.5f}, model={GetBestModelCTOR(model.estimator,model.best_params_)}", model.best_estimator_ 

def ClassificationReport(model, X_test, y_test, target_names=None):
    #assert X_test.shape[0]==y_test.shape[0]
    print("\nDetailed classification report:")
    print("\tThe model is trained on the full development set.")
    print("\tThe scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)                 
    print(classification_report(y_true, y_pred, target_names))
    print()
    
def FullReport(model, X_test, y_test, t):
    print(f"SEARCH TIME: {t:0.2f} sec")
    beststr, bestmodel = SearchReport(model)
    ClassificationReport(model, X_test, y_test)    
    print(f"CTOR for best model: {bestmodel}\n")
    print(f"{beststr}\n")
    return beststr, bestmodel
