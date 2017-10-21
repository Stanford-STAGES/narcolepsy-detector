# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:29:03 2017

@author: jens
"""

import numpy as np
import pickle
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import time
import scipy
from imblearn.over_sampling import SMOTE 
import pyGPs
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
def find_features_selected():
    P  = '/home/jens/Documents/stanford/feature_selection/'
    D = os.listdir(P)
    featuresT = np.zeros(489*2)
    
    for d in D:
        with open (P+d, 'rb') as fp:
            featSel = pickle.load(fp)

        featuresT += featSel['features']
        
        
    featureInd = featuresT[:489] + featuresT[489:]

    
    return featureInd
    
def load_data(name,train=True):
   
    with open (name, 'rb') as fp:
        D = pickle.load(fp)

    labels = D['labels']
    features = D['features'] 
    trainL = D['trainL']
    
    labels[labels==0] = -1

    if train:
        Ind = trainL!=0
    else:
        Ind = trainL==0
    
    labels = labels[Ind]
    features = features[Ind,:]
    trainL = trainL[Ind]
    
    return features,labels

if __name__ == '__main__':
    K = 5
    bestA = 8
    i = 8
    N_induction = 350
    P = '/home/jens/Documents/stanford/narco_features/'
    D = os.listdir(P)
    D.sort()
    D.append(D[0])
    featureIndT = find_features_selected()
    predictionList = list()    
    
    skf = StratifiedKFold(n_splits = K)

        
    first = False
    #D = D[4:]    
    #bestA = 8
    #i = 8
    for d in D:
        print(d)
        X, y = load_data(P+d)
        shuf = np.random.permutation(len(y))
        X = X[shuf,:]
        y = y[shuf]
        
        
        GPlist = list()
        
        if first:
            A = np.arange(3,12)
            trainScore = np.zeros([16,K])
            testScore = np.zeros([16,K])
        else:
            A = bestA
        
        predictions = np.zeros([len(y),4])        
        count = -1
        for train, test in skf.split(X, y):
            count += 1
            X_train = X[train,:]
            y_train = y[train]
            
            X_test = X[test,:]
            y_test = y[test]
            
            
            sm = SMOTE(k_neighbors = 10, m_neighbors = 20,n_jobs = -1)
            X_res, y_res = sm.fit_sample(X_train,y_train)            
            print(X_res.shape)
            
            #Set induction points
            inductionP = np.arange(len(y_train))
            np.random.shuffle(inductionP)
            inductionP = inductionP[:N_induction]
            inductionP = X_res[inductionP,:]
            
            
            featureInd = featureIndT>8
            print('Threshold: '+str(i)+ 'N. features: ' +str(sum(featureInd)))                
            L = np.ones(sum(featureInd))
            L = L.tolist()

            
            model = pyGPs.GPC_FITC()        

            #Set kernel and prior
            k = pyGPs.cov.RQard(log_ell_list=L, log_sigma=1.,log_alpha=0.0)
            m = pyGPs.mean.Zero()
            model.setPrior(mean=m, kernel=k, inducing_points=inductionP[:,featureInd])

            
            model.setData(X_res[:,featureInd],y_res)
            model.getPosterior()
            print("Negative log marginal likelihood before optimization:", round(model.nlZ,3))
            
            print('Optimization start... ')
            start = time.time()
            #GP.fit(X_res[:,featureInd],y_res)
            model.optimize()
            end = time.time()
            print("Negative log marginal likelihood optimized:", round(model.nlZ,3))
            print('Time elapsed: ' + str(end-start) + ' seconds')
            """
            trainSc = GP.score(X_train[:,featureInd],y_train)
            testSc = GP.score(X_test[:,featureInd],y_test)
            """
            pred = model.predict(X_test[:,featureInd])
            
            temp = pred[0]
            y_pred = np.ones(temp.shape)                
            y_pred[temp<0] = -1
            testSc = np.mean(np.squeeze(y_pred)==np.squeeze(y_test))
            print(testSc)
            
            #T = GP.predict_proba(X_test[:,featureInd])
            
            if first:
                testScore[i,count] = testSc
            else:
                predictions[test,0] = np.squeeze(pred[0])
                predictions[test,1] = np.squeeze(pred[1])                  
                predictions[test,2] = np.squeeze(pred[2])
                predictions[test,3] = np.squeeze(pred[3])
                
                GPlist.append(model)
                    
                   
        
        if first:
            first = False
            avPerformance = np.mean(testScore,axis=1)
            print(avPerformance)
            bestA = np.argmax(avPerformance)
        else:
            predictionList.append(predictions) 
            with open(d[:-4]+'_GPs', 'wb') as fp:
                pickle.dump(GPlist, fp)
        
        
    with open('GP_predictions', 'wb') as fp:
            pickle.dump(predictions, fp)
    
        