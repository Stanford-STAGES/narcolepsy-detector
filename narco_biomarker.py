# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:41:58 2017

@author: jens
"""

#import sc_test

import pyedflib
import scipy
import scipy.signal as signal
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sc_eval
import extract_features
#from scipy.fftpack import fft

class narco_biomarker(object):



    def __init__(self,config):
        self.config = config
        self.data_path = []
        self.fs = 100   
        self.fsH = 0.2
        self.fsL = 49
        self.CCsize =          {'C3':   2,
                                'C4':   2,
                                'O1':   2,
                                'O2':   2,
                                'EOG-L':4,
                                'EOG-R':4,
                                'EMG':  0.4,
                                'A1':   [],
                                'A2':   [],
                                 }
        
        self.channels = ['C3','C4','O1','O2','EOG-L','EOG-R','EMG','A1','A2']
        self.channels_used =   {'C3':   [],
                                'C4':   [],
                                'O1':   [],
                                'O2':   [],
                                'EOG-L':[],
                                'EOG-R':[],
                                'EMG':  [],
                                'A1':   [],
                                'A2':   [],
                                 }
 
        self.loaded_channels = {'C3':   [],
                                'C4':   [],
                                'O1':   [],
                                'O2':   [],
                                'EOG-L':[],
                                'EOG-R':[],
                                'EMG':  [],
                                'A1':   [],
                                'A2':   [],
                                 }
        self.encodedD = []
        self.f = []
        self.narco_features = []
        self.featureInd = []        
        self.narcolepsy_probability = []

    def loadEDF(self):
        if not self.f:
            self.f = pyedflib.EdfReader(self.data_path)

        for c in self.channels:
            if isinstance(self.channels_used[c], int ):
                 print(c)
                 self.loaded_channels[c] = self.f.readSignal(self.channels_used[c])

                 fs = self.f.samplefrequency(self.channels_used[c])
                 
                 self.filtering(c,fs)
                 self.resampling(c,fs)

                 
    def loadHeader(self):
        if not self.f:
            self.f = pyedflib.EdfReader(self.data_path)
            
        signal_labels = self.f.getSignalLabels()
        
        return signal_labels
        
    def filtering(self,c,fs):
        
        
        Fl = signal.butter(5, self.fsL/(fs/2), btype='lowpass', output='sos')
        Fh = signal.butter(5, self.fsH/(fs/2), btype='highpass', output='sos')
        
        self.loaded_channels[c] = signal.sosfiltfilt(Fl, self.loaded_channels[c], axis=-1, padtype='odd', padlen=None)
        self.loaded_channels[c] = signal.sosfiltfilt(Fh, self.loaded_channels[c], axis=-1, padtype='odd', padlen=None)
        
        
    def resampling(self,c,fs):
        
        ratio = np.round(np.float(self.fs))/np.round(np.float(fs));
        R = ratio.as_integer_ratio()

        self.loaded_channels[c] = signal.resample_poly(self.loaded_channels[c], R[0], R[1], axis=0, window=('kaiser', 5.0))
        
    def noise_level(self):
        
        
        name = '/home/jens/Documents/stanford/noiseM.mat'
        noiseM = sio.loadmat(name, squeeze_me = True, struct_as_record = False)
        noiseM = noiseM['noiseM']
        
        noise = np.ones(4)*np.inf
        count = -1
        for c in self.channels[:4]:
            count += 1

            if self.channels_used[c]:
                hjorth = self.extract_hjorth(self.loaded_channels[c])
                
                cov = np.array(noiseM.covM[count])
                
                covI = np.linalg.inv(cov)
                meanV = np.array(noiseM.meanV[count])
                noise[count] = np.sqrt(np.matmul(np.matmul(np.transpose(hjorth-meanV),covI),(hjorth-meanV)))
                
        notUsedC = np.argmax(noise[:2])         
        notUsedO = np.argmax(noise[2:4])+2
        
        self.channels_used[self.channels[notUsedC]] = []
        self.channels_used[self.channels[notUsedO]] = []
        
    def extract_hjorth(self,x):
        
        #Segment
        N = np.arange(0,len(x),100*60*5)
        B = np.array(zip(*(x[i:] for i in N)))
        
        #Activity
        act = np.var(B,axis=0)
        
        #Mobility
        
        mobil = self.mob(B)
        
        #Complexity
        comp = np.divide(self.mob(np.diff(B,axis=0)),mobil)
        
        #transform
        lAct = np.mean(np.log(act))
        lMobil = np.mean(np.log(mobil))
        lComp = np.mean(np.log(comp))
        
        
        return np.array([lAct,lMobil,lComp])
        
    def mob(self,B):
        diff = np.diff(B,axis=0)
        var = np.var(diff,axis=0)
        
        return np.sqrt(np.divide(var,np.var(B,axis=0)))
        
    def encoding(self):
        
        
                
        
        count = -1
        enc = [] 
        
        for c in self.channels:
            
            if isinstance(self.channels_used[c], int ):
                count += 1
                n  = int(self.fs*self.CCsize[c])
                p  = int(self.fs*(self.CCsize[c]-0.25))
                
                B1 = self.buffering(self.loaded_channels[c],n,p)
                
                zeroP = np.zeros([B1.shape[0]/2,B1.shape[1]])
                B1 = np.concatenate([zeroP,B1,zeroP],axis=0)
                
                                
                n  = int(self.fs*self.CCsize[c]*2)
                p  = int(self.fs*(self.CCsize[c]*2-0.25))
                
                B2 = self.buffering(np.concatenate([np.zeros(n/4),self.loaded_channels[c],np.zeros(n/4)]),
                                                             n,p)
                B2 = B2[:,:B1.shape[1]]
                
                F = np.fft.fft(B1,axis=0)
                
                    
                C = np.conj(np.fft.fft(B2,axis=0))                    
                    
                PS = np.multiply(F,C)
                CC = np.real(np.fft.fftshift(np.fft.ifft(PS,axis=0)))
                CC[np.isnan(CC)] = 0
                CC[np.isinf(CC)] = 0
                
                CC = CC[CC.shape[0]/4:CC.shape[0]*3/4,:]
                sc = np.max(CC,axis=0)
                sc = np.multiply(np.sign(sc),np.log(np.abs(sc)+1))/(sc+1e-10)
                
                CC = np.multiply(CC,sc)
                CC.astype(np.float32)
                
                if len(enc)>0:
                    enc = np.concatenate([enc,CC])
                else:
                    enc = CC
                
                if count==2:
                    eog1 = F
                if count==3:
                    PS = eog1*C
                    CC = np.real(np.fft.fftshift(np.fft.ifft(PS,axis=0)))
                    CC = CC[CC.shape[0]/4:CC.shape[0]*3/4,:]
                    sc = np.max(CC,axis=0)
                    sc = np.multiply(np.sign(sc),np.log(np.abs(sc)+1))/(sc+1e-10)
                
                    CC = np.multiply(CC,sc)
                    CC.astype(np.float32)
                    
                    enc = np.concatenate([enc,CC])
        
        self.encodedD = enc
        
        
    def buffering(self,x, n, p=0):
        
        if p >= n:
            raise ValueError('p ({}) must be less than n ({}).'.format(p,n))
    
        # Calculate number of columns of buffer array
        cols = int(np.floor(len(x)/(n-p)))
        # Check for opt parameters

    
        # Create empty buffer array
        b = np.zeros((n, cols))
    
        # Fill buffer by column handling for initial condition and overlap
        j = 0
        slide = n - p
        start = 0
        for i in range(cols-int(np.ceil(n/(n-p)))):
            # Set first column to n values from x, move to next iteration
        
            b[:,i] = x[start:start+n]
            start += slide
                        
    
        return b
    def score_data(self):
        
        self.hypnodensity = list()
        for l in self.config.models_used:
            hyp = sc_eval.run_data(self.encodedD,l)
            self.hypnodensity.append(hyp)
            
    def get_hypnodensity(self):
        
        
        print(self.hypnodensity)
        av = np.zeros(self.hypnodensity[0].shape)
        
        for i in range(len(self.hypnodensity)):
            av += self.hypnodensity[i]
            
        av = np.divide(av,len(self.hypnodensity))
        return av
        
    def eval_GP(self):
        print('Do GP!')
        count = -1
        valueM = np.zeros([len(self.config.models_used),5])
        valueV = np.zeros([len(self.config.models_used),5])
        
        for l in self.config.models_used:
            count += 1
            narcoFeatures = extract_features.extract(self.hypnodensity[count])
            narcoFeatures = np.expand_dims(narcoFeatures,axis=1)

            with open(l+'_GPs', 'rb') as fp:
                GPList = pickle.load(fp)               
        
            for i in range(5):
                model = GPList[i]
                
                T = model.predict(narcoFeatures.T)
                valueM[count,i] = T[0]
                valueV[count,i] = T[1]
                                
        P = np.multiply(valueM,valueV)
        print(P.shape)
        P = np.sum(P)/np.sum(valueV)
        
        self.narcolepsy_probability = P
        
        
    def plotHypnogram(self):
        
        print('plot it')
        
    def eval_all(self):
        
        self.loadEDF()
        self.noise_level()
        self.encoding()
        
        self.score_data()

        self.eval_GP()
        