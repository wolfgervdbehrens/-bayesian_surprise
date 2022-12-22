"""
Gwendolyn English 10.07.2022

Functions to determine whether drift is present between protocols. 
"""

##########################################################################################################
#Import required packages & functions
import pickle
import sys
import os
import numpy as np
import pandas as pd  
from scipy import stats
from scipy import signal 

from plotting import * 
from helper_functions import *
##########################################################################################################

def pullDataPreProbe_MUA(filepath, shank):
    """
    This function reads the compiled data file and extracts relevant data required for specified analysis.
    Inputs: filepath, shank (Shank1, Shank0)
    Outputs: Pandas dataframe with corresponding data
    """
    #Load data
    data = pickle.load(open(filepath, 'rb'))
    time = data['time']
    del data['time']
    df = pd.DataFrame(data)
    
    #Establish trials to pull, predetermined to investigate C2 response
    extData = df[df['whiskerID'] == 'C2']
    df = extData
    
    #Establish shank to pull
    if shank == 0:
        extData = df[df['shank'] == 0]
    if shank == 1:
        extData = df[df['shank'] == 1]
    df = extData
    
    #Seperate Pattern from Random data 
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']
            
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    return (df1, df2)


def trialbytrial_preprobe_MUA(df1, df2, name, region, path):    
    #Trial-by-trial ttest holders
    fr_ttest = []
    fr_P = []
    fr_R = [] 
    
    fullFR_P = []
    fullFR_R = [] 
    
    layer = []
    animalID = []
    
    #Create MUA Analysis Folder
    if not os.path.exists(path + '/Analysis/Control'):
        os.mkdir(path + '/Analysis/Control')
    
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        layer.append(df1.loc[row, 'layer'])
        animalID.append(df1.loc[row, 'animalID'])
        
        #Extract specific pre-probe
        if region == "cortex":
            indices = np.arange(18,1320,33)
        if region == "thalamus":
            indices = np.arange(19,1280,33)
        inputdata1 = np.asarray(df1.loc[row, 'ms_bins'])[indices,:]
        inputdata2 = np.asarray(df2.loc[row, 'ms_bins'])[indices,:]
        
        #convert to firing rates and extract peaks 
        data1 = np.amax(inputdata1, axis = 1)
        data2 = np.amax(inputdata2, axis = 1)
        
        #Remove np.nan entries
        data1nans = np.argwhere(np.isnan(data1)).flatten()
        data2nans = np.argwhere(np.isnan(data2)).flatten()
        naninds = np.hstack((data1nans, data2nans)).flatten()
        naninds = naninds.astype(int)
        data1 = np.delete(data1, naninds)
        data2 = np.delete(data2, naninds)
        
        #Relative ttest
        tstat, pval = stats.ttest_rel(data1, data2) 
        
        #Append results
        fr_ttest.append(pval)
        fr_P.append(np.mean(data1))
        fr_R.append(np.mean(data2))
        
        #Extract full ms bins and average per channel 
        datafullFR1 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(inputdata1), axis = 0)
        datafullFR2 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(inputdata2), axis = 0)
        
        #Append results
        fullFR_P.append(datafullFR1)
        fullFR_R.append(datafullFR2)
        
    #Lists to arrays
    fr_P = np.asarray(fr_P) 
    fr_R = np.asarray(fr_R) 
    fr_ttest = np.asarray(fr_ttest)
    layer = np.asarray(layer) 
    
    fullFR_P = np.asarray(fullFR_P)
    fullFR_P = np.asarray(fullFR_P)
    
    #FDR Corrected 
    corrected_sig, bh_val = benjamini_hochberg(fr_ttest, Q = 0.05)
    
    #Plot Max Magnitude Response Scatters
    scatter2D_edgelabels_staticAxes(fr_P,fr_R,fr_ttest, path + '/Analysis/Control/' + \
              'PreProbe_Control_Scatter_FiringRate_mean_'+ name + '.png','Control', corrected_sig)
    
    #Write data to csv    
    datatofile = np.array([animalID, layer, fr_P, fr_R, fr_ttest, corrected_sig, bh_val])
    datatofile = datatofile.T
    
    headers = np.array(['Animal', 'Layer', 'FR_P_mean', 'FR_R_mean','p_Val', 'Sig_afterCorr', 'BH_value'])
    
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/Control/' + 'PreProbe_Control_FiringRate_ttestResults_' \
                                    + name + '.csv', header = headers)