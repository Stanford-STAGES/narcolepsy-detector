# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:18:56 2017

@author: jens
"""
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
from sklearn.feature_selection import RFECV
from narcolepsy_GP import load_data
import multiprocessing

def run_selection(n):
    
    P = '/home/jens/Documents/stanford/narco_features/'
    S = '/home/jens/Documents/stanford/feature_selection/'
    D = os.listdir(P)
    D.sort()    
    print(P+D[n])
    
    features,labels = load_data(P+D[n])  
    
    print(features.shape)
    
    features2 = np.square(features)
    
    X = np.concatenate([features,features2],axis=1)
    y = labels
    skf = StratifiedKFold(n_splits = 20)
    
    svc = SVC(kernel="linear")
    
    rfecv = RFECV(estimator=svc, step=1, cv=skf,
                  scoring='accuracy')
    
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    relevant = rfecv.get_support()    
    score = max(rfecv.grid_scores_)
    
    selFeatures = {'features': relevant,
                        'score': score}
                        
    with open(S+D[n], 'wb') as fp:
        pickle.dump(selFeatures, fp)
        
if __name__ == '__main__':
    
    for i in range(16):
        run_selection(i)    
    
        
