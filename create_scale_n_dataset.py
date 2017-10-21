# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:52:36 2017

@author: jens
"""

import os
import scipy.io as sio
import extract_features
import numpy as np
import matplotlib.pyplot as plt    
import csv    
import pickle

def load_csv():   
    N = '/home/jens/Documents/stanford/overview_file_cohorts.csv'
    trainL = []
    label = []
    ID = []
    with open(N) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:        
            
            trainL += [int(row['Used for narco training'])]
            ID += [row['ID']]
            
            label += [int(row['Label'])]
            
            
    return ID, label, trainL
    
if __name__ == '__main__':
    ID, labelT, trainLT = load_csv()
    
    P = '/home/jens/Documents/stanford/scored_data/'
    D = os.listdir(P)
    D.sort()
    
    
    for d in D:
        F = os.listdir(P+d)
        F.sort()        
        count = 0        
        labels = np.zeros(len(F))
        trainL = np.zeros(len(F))
        featStack = []

        for f in F:
            name = f[30:-7]
            try:
                index = ID.index(name)
            except:
                print(name + ' not found')
                continue

            print(str(count) + '/' + str(len(F)))
            contents = sio.loadmat(P+d+'/'+f)
            pred = contents['predictions']

            if len(pred)==0:
                continue
            
            
            labels[count] = labelT[index]
            trainL[count] = trainLT[index]
            
            feat = extract_features.extract(pred)
            feat = np.expand_dims(feat,axis=1)
            if len(featStack)==0:
                featStack = feat
            else:
                featStack = np.concatenate([featStack,feat],axis=1)
                
            count += 1
        featStack = np.transpose(featStack)
        labels = labels[:count]
        trainL = trainL[:count]
        m = np.mean(featStack,axis=1)+1e-10
        v = np.percentile(featStack,85,axis=1) - np.percentile(featStack,15,axis=1)+1e-10
        
        m = np.expand_dims(m,axis=1)
        featStackScaled = featStack/np.expand_dims(v,axis=1)
        
        featStackScaled[10<featStackScaled] = 10
        featStackScaled[-10>featStackScaled] = -10
        
        data = {'features': featStackScaled,
                'labels': labels,
                'trainL': trainL,
                }
                
        scale = {'mean': m,
                 'range': v}
                 
        output = open('narco_features/' + d +'_narcodata.pkl', 'wb')
        pickle.dump(data, output, -1)
        output.close()
