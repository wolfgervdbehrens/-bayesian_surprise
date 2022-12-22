"""
Gwendolyn English 17.11.2021

Functions for the wavelet analysis between all animals within an experimental set 
"""

##########################################################################################################
#Import required packages & functions
import sys
import os
import numpy as np
import pandas as pd  
from scipy import stats
from scipy import signal 
import math
import matplotlib.pyplot as plt

from plotting import * 
from helper_functions import *
##########################################################################################################

def pullDataWave(filepath, trials, shank):
    """
    This function reads the compiled data file and extracts relevant data required for specified analysis.
    Inputs: filepath, Trials ('probe', 'context'), shank (0,1)
    Outputs: Pandas dataframe with corresponding data
    """
    #Load data
    data = pickle.load(open(filepath, 'rb'))

    #Select trial-by-trial data for dataframe
    trialData = {'animalID': data['animalID'], 'shank': data['shank'], 'layer': data['layer'],
                 'trode': data['trode'], 'paradigm': data['paradigm'], 'ketID': data['ketID'],
                 'whiskerID': data['whiskerID'], 'ERP': data['ERP'], 'PSD': data['PSD'], 'prestimPSD': data['prestimPSD']}
    df = pd.DataFrame(data = trialData)

    #Establish trials to pull
    if trials == 'probe':
        extData = df[df['whiskerID'] == 'C1']         
    if trials == 'context':
        extData = df[df['whiskerID'].isin(['B1','C2','D1'])]  
    df = extData    
    
    #Establish shank to pull
    if shank == 0:
        extData = df[df['shank'] == 0]         
    if shank == 1:
        extData = df[df['shank'] == 1]   
    df = extData   
    
    #Separate Pattern and Random Data 
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']        

    df1 = df1.reset_index(drop = True)
    df2 = df2.reset_index(drop = True)

    return df1, df2


def trialbytrialWave(path, df1, df2, name, stim):    
    
    #Create Holders
    wavelet_P = []
    wavelet_R = [] 
    wavelet_diff = []
    
    animal = []
    layer = [] 
    
    #Set wavelet parameters
    w = 6 #Constant 
    fs = 1e3 #Sample rate (ERP signal is downsampled from 32k to 1k Hz) 
    freq = np.linspace(4,60,57)  #Evalutate all resolvable frequencies up to gamma band 
    widths = w*fs / (2*freq*np.pi)
    t = np.linspace(-50,250,300)
    
    #Create Analysis Folder
    if not os.path.exists(path + '/Analysis/TF'):
        os.mkdir(path + '/Analysis/TF')
    
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        
        #Append label data
        layer.append(df1.loc[row, 'layer'])
        animal.append(df1.loc[row, 'animalID'])        
        currentanimal = df1.loc[row, 'layer']
        currentlayer = df1.loc[row, 'animalID']
        
        #Select ERP signal data 
        data1 = np.asarray(df1.loc[row, 'ERP'])
        data2 = np.asarray(df2.loc[row, 'ERP'])

        #Remove any nan entries
        data1nans = np.argwhere(np.isnan(data1[:,0])).flatten()
        data2nans = np.argwhere(np.isnan(data2[:,0])).flatten()
        naninds = np.hstack((data1nans, data2nans)).flatten()
        naninds = naninds.astype(int)
        data1 = np.delete(data1, naninds, axis = 0)
        data2 = np.delete(data2, naninds, axis = 0)
        
        #Holder arrays for trial by trial wavelet results     
        WaveletTrial_P = []
        WaveletTrial_R = []
        
        #Cycle through all trials, completing wavelets for both pattern and random protocols 
        for trial in np.arange(np.shape(data1)[0]):
            #Pattern
            current_Trial_P = data1[trial,:]
            cwtmP = signal.cwt(current_Trial_P, signal.morlet2, widths, w = w)
            WaveletTrial_P.append(cwtmP)
            
            #Random
            current_Trial_R = data2[trial,:]
            cwtmR = signal.cwt(current_Trial_R, signal.morlet2, widths, w = w)
            WaveletTrial_R.append(cwtmR)

        #Calculate average wavelet for current channel and append to overall holder
        avg_WaveletTrial_P = np.mean(np.asarray(WaveletTrial_P), axis = 0)
        wavelet_P.append(avg_WaveletTrial_P)
        
        avg_WaveletTrial_R = np.mean(np.asarray(WaveletTrial_R), axis = 0)
        wavelet_R.append(avg_WaveletTrial_R)
       
        WaveletTrial_diff = np.abs(avg_WaveletTrial_P) - np.abs(avg_WaveletTrial_R)
        wavelet_diff.append(WaveletTrial_diff)
     
    #Lists to arrays
    animal = np.asarray(animal)
    layer = np.asarray(layer) 
    compLayer = laminar_labelTolayer(layer)
    
    #Arrays to dataframe for easy access 
    df = pd.DataFrame({'wavelet_P': wavelet_P, 'wavelet_R': wavelet_R, 'wavelet_diff': wavelet_diff,\
                       'compLayer': compLayer, 'animal':animal})   
    
    #Supragranluar Layer  
    SG_P = np.mean(np.asarray(df[df['compLayer']=='SG']['wavelet_P']), axis = 0)
    SG_R = np.mean(np.asarray(df[df['compLayer']=='SG']['wavelet_R']), axis = 0)    
    SG_diff = np.abs(SG_P) - np.abs(SG_R)
    maxVal = np.max(np.abs(SG_P) + np.abs(SG_R))
    SG_nis = (np.abs(SG_P) - np.abs(SG_R))/(np.abs(SG_R) + np.abs(SG_P))
    SG_nis = SG_nis * ((np.abs(SG_R) + np.abs(SG_P))/ maxVal) #Normalization factor 
     
    plot_wavelet_diff_scales(t, freq, SG_nis, [-0.04, 0.04], \
                             path + '/Analysis/TF/WaveletDiff_SG_setScale_' +name+ '_'+stim+'.png')
    
    
    #Granluar Layer  
    G_P = np.mean(np.asarray(df[df['compLayer']=='G']['wavelet_P']), axis = 0)
    G_R = np.mean(np.asarray(df[df['compLayer']=='G']['wavelet_R']), axis = 0)    
    G_diff = np.abs(G_P) - np.abs(G_R)
    maxVal = np.max(np.abs(G_P) + np.abs(G_R))
    G_nis = (np.abs(G_P) - np.abs(G_R))/(np.abs(G_R) + np.abs(G_P))
    G_nis = G_nis * ((np.abs(G_R) + np.abs(G_P))/ maxVal) #Normalization factor 
    
    plot_wavelet_diff_scales(t, freq, G_nis, [-0.04, 0.04], \
                             path + '/Analysis/TF/WaveletDiff_G_setScale_' +name+ '_'+stim+'.png')
   
    
    #Infragranluar Upper Layer  
    IGU_P = np.mean(np.asarray(df[df['compLayer']=='IGU']['wavelet_P']), axis = 0)
    IGU_R = np.mean(np.asarray(df[df['compLayer']=='IGU']['wavelet_R']), axis = 0)    
    IGU_diff = np.abs(IGU_P) - np.abs(IGU_R)
    maxVal = np.max(np.abs(IGU_P) + np.abs(IGU_R))
    IGU_nis = (np.abs(IGU_P) - np.abs(IGU_R))/(np.abs(IGU_R) + np.abs(IGU_P))
    IGU_nis = IGU_nis * ((np.abs(IGU_R) + np.abs(IGU_P))/ maxVal) #Normalization factor 
    
    plot_wavelet_diff_scales(t, freq, IGU_nis, [-0.04, 0.04], \
                             path + '/Analysis/TF/WaveletDiff_IGU_setScale_' +name+ '_'+stim+'.png')
    
    #Infragranluar Lower Layer  
    IGL_P = np.mean(np.asarray(df[df['compLayer']=='IGL']['wavelet_P']), axis = 0)
    IGL_R = np.mean(np.asarray(df[df['compLayer']=='IGL']['wavelet_R']), axis = 0)    
    IGL_diff = np.abs(IGL_P) - np.abs(IGL_R)
    maxVal = np.max(np.abs(IGL_P) + np.abs(IGL_R))
    IGL_nis = (np.abs(IGL_P) - np.abs(IGL_R))/(np.abs(IGL_R) + np.abs(IGL_P))
    IGL_nis = IGL_nis * ((np.abs(IGL_R) + np.abs(IGL_P))/ maxVal) #Normalization factor 
    
    plot_wavelet_diff_scales(t, freq, IGL_nis, [-0.04, 0.04], \
                             path + '/Analysis/TF/WaveletDiff_IGL_setScale_' +name+ '_'+stim+'.png')
    