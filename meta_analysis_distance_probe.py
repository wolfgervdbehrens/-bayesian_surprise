"""
Gwendolyn English 28.07.2022

Functions for the PSD Distance from Probe analysis between all animals within an experimental set. 
"""

##########################################################################################################
#Import required packages & functions
import sys
import os
import numpy as np
import pandas as pd  
from scipy import stats
from scipy import signal 
import scikit_posthocs as sp

from plotting import * 
from helper_functions import *
##########################################################################################################

def pullDataDistProbe(filepath, shank):
    """
    This function reads the compiled data file and extracts relevant data required for specified analysis.
    Inputs: filename, shank (0,1)
    Outputs: Pandas dataframe with corresponding data
    """
    #Load data
    data = pickle.load(open(filepath, 'rb'))

    preFreqs = data['prestimPSD_freqs']
    postFreqs = data['PSD_freqs']
    
    #Select trial-by-trial data for dataframe
    trialData = {'animalID': data['animalID'], 'shank': data['shank'], 'layer': data['layer'],
                 'trode': data['trode'], 'paradigm': data['paradigm'], 'ketID': data['ketID'],
                 'whiskerID': data['whiskerID'], 'ERP': data['ERP'], 'PSD': data['PSD'], 'prestimPSD': data['prestimPSD']}
    df = pd.DataFrame(data = trialData)
    
    #Select standard PSD data
    extData = df[df['whiskerID'].isin(['B1','C2','D1'])]  
    df = extData 
      
    #Establish shank to pull
    if shank == 0:
        extData = df[df['shank'] == 0]         
    if shank == 1:
        extData = df[df['shank'] == 1]   
    df = extData   
    
    #Separate Pattern and Random data 
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']
           
    df1 = df1.reset_index(drop = True)
    df2 = df2.reset_index(drop = True)

    return df1, df2, preFreqs, postFreqs

def freqbyfreq(path, df1, df2, preFreqs, postFreqs, name):    
    
#####Create holder arrays
    animal = []
    layer = [] 
    paradigm = []    
    stim = []
    freq = []
    KWpval = []
    dunnPosthoc = [] 
    
    #Mean PSD values
    preStim = []
    ms250 = []
    ms500 = []
    ms750 = []
    ms1000 = []
    s12 = []
    s23 = []
    s34 =[]
    s45 = []
    s5p = []
    
    #Create Analysis Folder for copious plots
    if not os.path.exists(path + '/Analysis/DistancefromProbe'):
        os.mkdir(path + '/Analysis/DistancefromProbe')
    
####Distance from Probe Indices for each context whisker in Pattern and Random protocols
    pattern_B1 = np.tile(np.array([3,6,9,12,15,18,21,24,27,30,33,2,5,8,11,14,17,20,23,\
                                        3,6,9,12,15,18,21,24,27,1,4,7,10]), 40)
    random_B1 = np.tile(np.array([2,4,9,10,15,18,20,23,27,28,32,5,6,11,14,16,17,19,24, \
                                        2,4,9,10,15,18,20,23,28,3,5,10,11]), 40)
    pattern_C2 = np.tile(np.array([1,4,7,10,13,16,19,22,25,28,31,3,6,9,12,15,18,21,24,\
                                                       1,4,7,10,13,16,19,22,25,28,2,5,8,11]), 40)
    random_C2 = np.tile(np.array([1,6,8,11,14,16,21,22,26,30,31,2,4,7,10,12,18,22,25,\
                                                      1,6,8,11,14,16,21,22,24,26,1,2,7,9]), 40)
    pattern_D1 = np.tile(np.array([2,5,8,11,14,17,20,23,26,29,32,1,4,7,10,13,16,19,22,25, \
                                                       2,5,8,11,14,17,20,23,26,3,6,9]), 40)
    random_D1 = np.tile(np.array([3,5,7,12,13,17,19,24,25,29,33,1,3,8,9,13,15,20,21,23,\
                                                      3,5,7,12,13,17,19,25,27,4,6,8]), 40)
    
####Loop through all site entries     
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        
        #Identify current whisker stim and other meta info 
        currentAnimal = df1.loc[row, 'animalID']
        currentLayer = df1.loc[row, 'layer']
        currentStim = df1.loc[row, 'whiskerID']
        if currentStim == 'B1': 
            stimSeq_P = pattern_B1
            stimSeq_R = random_B1
        if currentStim == 'C2': 
            stimSeq_P = pattern_C2
            stimSeq_R = random_C2   
        if currentStim == 'D1': 
            stimSeq_P = pattern_D1
            stimSeq_R = random_D1         

#########Extract Data Pattern; Sorted Distance indices
        dataP = df1.loc[row, 'PSD']
        dataPpre = df1.loc[row, 'prestimPSD']      

        dist1_pattern = np.asarray(np.where(stimSeq_P == 1)).flatten()
        dist2_pattern = np.asarray(np.where(stimSeq_P == 2)).flatten()
        dist3_pattern = np.asarray(np.where(stimSeq_P == 3)).flatten()
        dist4_pattern = np.asarray(np.where(stimSeq_P == 4)).flatten()
        dist5_8_pattern = np.asarray(np.where(np.in1d(stimSeq_P, [5,6,7,8]))).flatten() 
        dist9_12_pattern = np.asarray(np.where(np.in1d(stimSeq_P, [9,10,11,12]))).flatten()
        dist13_21_pattern = np.asarray(np.where(np.in1d(stimSeq_P, \
                                                                [13,14,15,16,17,18,19,20,21]))).flatten()
        dist22_30_pattern = np.asarray(np.where(np.in1d(stimSeq_P, \
                                                                [22,23,24,25,26,27,28,29,30]))).flatten()
        dist31_plus_pattern = np.asarray(np.where(stimSeq_P >= 31)).flatten()
        
#########Extract Data Random; Sorted Distance indices
        dataR = df2.loc[row, 'PSD']
        dataRpre = df2.loc[row, 'prestimPSD']
        
        #Sorted Distance Indices Random
        dist1_random = np.asarray(np.where(stimSeq_R == 1)).flatten()
        dist2_random = np.asarray(np.where(stimSeq_R == 2)).flatten()
        dist3_random = np.asarray(np.where(stimSeq_R == 3)).flatten()
        dist4_random = np.asarray(np.where(stimSeq_R == 4)).flatten()
        dist5_8_random = np.asarray(np.where(np.in1d(stimSeq_R, [5,6,7,8]))).flatten() 
        dist9_12_random = np.asarray(np.where(np.in1d(stimSeq_R, [9,10,11,12]))).flatten()
        dist13_21_random = np.asarray(np.where(np.in1d(stimSeq_R, \
                                                                [13,14,15,16,17,18,19,20,21]))).flatten()
        dist22_30_random = np.asarray(np.where(np.in1d(stimSeq_R, \
                                                                [22,23,24,25,26,27,28,29,30]))).flatten()
        dist31_plus_random = np.asarray(np.where(stimSeq_R >= 31)).flatten()
        
########Cycle through frequencies, extract sorted data, complete tests
        for freq_bin in np.linspace(8,100,24):
            
            #Identify relevant column in pre-and post-stimulus PSD data and extract data 
            columnPre, = np.where(preFreqs == freq_bin)
            columnPost, = np.where(postFreqs == freq_bin)
            
            #Determine if there is a PSD entry for current frequency in the PSD 
            if columnPre.size: 
                pre = 1 
            else:
                pre = 0
                
            #Extract correct column 
            currentDataPpre = np.take(dataPpre, columnPre, axis = 1)
            currentDataP = np.take(dataP, columnPost, axis = 1)
            currentDataRpre = np.take(dataRpre, columnPre, axis = 1)
            currentDataR = np.take(dataR, columnPost, axis = 1)
            
############Pattern
            #Append meta info 
            freq.append(freq_bin)
            layer.append(currentLayer)
            animal.append(currentAnimal) 
            stim.append(currentStim)
            paradigm.append('P')
            
            #Extract sorted PSDs 
            dist1_psdVals = np.take(currentDataP, dist1_pattern) 
            dist2_psdVals = np.take(currentDataP, dist2_pattern) 
            dist3_psdVals = np.take(currentDataP, dist3_pattern) 
            dist4_psdVals = np.take(currentDataP, dist4_pattern) 
            dist5_8_psdVals = np.take(currentDataP, dist5_8_pattern)             
            dist9_12_psdVals = np.take(currentDataP, dist9_12_pattern) 
            dist13_21_psdVals = np.take(currentDataP, dist13_21_pattern)  
            dist22_30_psdVals = np.take(currentDataP, dist22_30_pattern) 
            dist31_psdVals = np.take(currentDataP, dist31_plus_pattern) 
            
            
            #Remove nan entries 
            dist1_psdVals = dist1_psdVals[~np.isnan(dist1_psdVals)]
            dist2_psdVals = dist2_psdVals[~np.isnan(dist2_psdVals)]
            dist3_psdVals = dist3_psdVals[~np.isnan(dist3_psdVals)]
            dist4_psdVals = dist4_psdVals[~np.isnan(dist4_psdVals)]
            dist5_8_psdVals = dist5_8_psdVals[~np.isnan(dist5_8_psdVals)]
            dist9_12_psdVals = dist9_12_psdVals[~np.isnan(dist9_12_psdVals)]
            dist13_21_psdVals = dist13_21_psdVals[~np.isnan(dist13_21_psdVals)]
            dist22_30_psdVals = dist22_30_psdVals[~np.isnan(dist22_30_psdVals)]
            dist31_psdVals = dist31_psdVals[~np.isnan(dist31_psdVals)]
            
            #Prestimulus where available
            if pre == 1:
                dist1_psdVals_pre = np.take(currentDataPpre, dist1_pattern)
                #Remove nan entries
                dist1_psdVals_pre = dist1_psdVals_pre[~np.isnan(dist1_psdVals_pre)]
            
            #Append sorted PSD means to meta holders
            if pre == 1: preStim.append(np.mean(dist1_psdVals_pre))
            else: preStim.append(np.nan)
            ms250.append(np.mean(dist1_psdVals))
            ms500.append(np.mean(dist2_psdVals))
            ms750.append(np.mean(dist3_psdVals))
            ms1000.append(np.mean(dist4_psdVals))
            s12.append(np.mean(dist5_8_psdVals))
            s23.append(np.mean(dist9_12_psdVals))
            s34.append(np.mean(dist13_21_psdVals))
            s45.append(np.mean(dist22_30_psdVals))
            s5p.append(np.mean(dist31_psdVals))
            
            #Check for significance between groups with Kruskal-Wallis H test 
            h_stat, p_val = stats.kruskal(dist1_psdVals, dist2_psdVals, dist3_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)
            KWpval.append(p_val)
            
            #Compile data for plotting and Dunn posthoc-test
            datatoPlot = [dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            
            #Complete Dunn tests and append 
            if p_val <0.05:
                dunn_vals = sp.posthoc_dunn(datatoPlot)
                dunnPosthoc.append(np.hstack((np.asarray(dunn_vals).flatten(), np.zeros(64))))
            else: dunnPosthoc.append(np.zeros(145)) #81 for pattern, 64 for random   
            
            #If significant and pre-stimulus time also available, print plot with prestim vals 
            if p_val < 0.05 and pre ==1: 
                datatoPlot = [dist1_psdVals_pre, dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
                figureTitle = str(currentLayer) + ' ' + str(freq_bin) + ' Hz'
                figureName = 'Pattern_' + str(currentAnimal) +'_' + name + '_' + str(currentLayer) + '_' + \
                str(currentStim) +  '_' + str(freq_bin) + 'Hz' + '_' + str(np.around(p_val,4)) +'_pre.png'
                dataTitles = np.array(['200ms','250ms','500ms','750ms','1s','1-2s', '2-3s','3-4s','4-5s','5+s'])
                plot_boxplot(datatoPlot, dataTitles, figureTitle, figureName,  \
                                                path + '/Analysis/DistancefromProbe') 
            
            #Clear PSD Value arrays 
            dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, dist5_8_psdVals,\
            dist9_12_psdVals, dist13_21_psdVals,dist22_30_psdVals, dist31_psdVals = None, None, None, None, \
            None, None, None, None, None
            
#############Random
            #Append meta info 
            freq.append(freq_bin)
            layer.append(currentLayer)
            animal.append(currentAnimal) 
            stim.append(currentStim)
            paradigm.append('R')
            
            #Extract sorted PSDs 
            dist1_psdVals = np.take(currentDataR, dist1_random) 
            dist2_psdVals = np.take(currentDataR, dist2_random) 
            dist3_psdVals = np.take(currentDataR, dist3_random) 
            dist4_psdVals = np.take(currentDataR, dist4_random) 
            dist5_8_psdVals = np.take(currentDataR, dist5_8_random)             
            dist9_12_psdVals = np.take(currentDataR, dist9_12_random) 
            dist13_21_psdVals = np.take(currentDataR, dist13_21_random)  
            dist22_30_psdVals = np.take(currentDataR, dist22_30_random) 
            dist31_psdVals = np.take(currentDataR, dist31_plus_random) 
            
            #Remove nan entries 
            dist1_psdVals = dist1_psdVals[~np.isnan(dist1_psdVals)]
            dist2_psdVals = dist2_psdVals[~np.isnan(dist2_psdVals)]
            dist3_psdVals = dist3_psdVals[~np.isnan(dist3_psdVals)]
            dist4_psdVals = dist4_psdVals[~np.isnan(dist4_psdVals)]
            dist5_8_psdVals = dist5_8_psdVals[~np.isnan(dist5_8_psdVals)]
            dist9_12_psdVals = dist9_12_psdVals[~np.isnan(dist9_12_psdVals)]
            dist13_21_psdVals = dist13_21_psdVals[~np.isnan(dist13_21_psdVals)]
            dist22_30_psdVals = dist22_30_psdVals[~np.isnan(dist22_30_psdVals)]
            dist31_psdVals = dist31_psdVals[~np.isnan(dist31_psdVals)]
            
            #Prestimulus where available
            if pre == 1:
                dist1_psdVals_pre = np.take(currentDataRpre, dist1_pattern)
                #Remove nan entries
                dist1_psdVals_pre = dist1_psdVals_pre[~np.isnan(dist1_psdVals_pre)]
            
            #Append sorted PSD means and medians to meta holders
            if pre == 1: preStim.append(np.mean(dist1_psdVals_pre))
            else: preStim.append(np.nan)
            ms250.append(np.mean(dist1_psdVals))
            ms500.append(np.mean(dist2_psdVals))
            ms750.append(np.mean(dist3_psdVals))
            ms1000.append(np.mean(dist4_psdVals))
            s12.append(np.mean(dist5_8_psdVals))
            s23.append(np.mean(dist9_12_psdVals))
            s34.append(np.mean(dist13_21_psdVals))
            s45.append(np.mean(dist22_30_psdVals))
            s5p.append(np.mean(dist31_psdVals))
            
            #Check for significance between groups with Kruskal-Wallis H test, here, accounting for missing data
            if currentStim == 'B1':
                h_stat, p_val = stats.kruskal(dist2_psdVals, dist3_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)
                datatoPlot = [dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            if currentStim == 'C2':
                h_stat, p_val = stats.kruskal(dist1_psdVals, dist2_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)
                datatoPlot = [dist1_psdVals, dist2_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            if currentStim == 'D1':
                h_stat, p_val = stats.kruskal(dist1_psdVals, dist3_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)    
                datatoPlot = [dist1_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            KWpval.append(p_val)

            #Complete Dunn tests and append 
            if p_val <0.05:
                dunn_vals = sp.posthoc_dunn(datatoPlot)
                dunnPosthoc.append(np.hstack((np.zeros(81), np.asarray(dunn_vals).flatten())))
            else: dunnPosthoc.append(np.zeros(145)) 
            
            #Recollect all data for plotting
            datatoPlot = [dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            
            #If significant and pre-stimulus time also available, print plot with prestim vals 
            if p_val < 0.05 and pre ==1: 
                datatoPlot = [dist1_psdVals_pre, dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
                figureTitle = str(currentLayer) + ' ' + str(freq_bin) + ' Hz'
                figureName = 'Random_' + str(currentAnimal) +'_' + name + '_' + str(currentLayer) + '_' + \
                str(currentStim) +  '_' + str(freq_bin) + 'Hz' + '_' + str(np.around(p_val,4)) +'_pre.png'
                dataTitles = np.array(['200ms', '250ms','500ms','750ms','1s','1-2s','2-3s','3-4s','4-5s','5+s'])
                plot_boxplot(datatoPlot, dataTitles, figureTitle, figureName,  \
                                                 path + '/Analysis/DistancefromProbe')   
            
            #Clear PSD Value arrays 
            dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, dist5_8_psdVals,\
            dist9_12_psdVals, dist13_21_psdVals,dist22_30_psdVals, dist31_psdVals = None, None, None, None, \
            None, None, None, None, None
            
############Pattern Corrected to align to Random protocol 
            #Append meta info 
            freq.append(freq_bin)
            layer.append(currentLayer)
            animal.append(currentAnimal) 
            stim.append(currentStim)
            paradigm.append('P_Corr')
            
            #Extract sorted PSDs 
            dist1_psdVals = np.take(currentDataP, dist1_pattern) 
            dist2_psdVals = np.take(currentDataP, dist2_pattern) 
            dist3_psdVals = np.take(currentDataP, dist3_pattern) 
            dist4_psdVals = np.take(currentDataP, dist4_pattern) 
            dist5_8_psdVals = np.take(currentDataP, dist5_8_pattern)             
            dist9_12_psdVals = np.take(currentDataP, dist9_12_pattern) 
            dist13_21_psdVals = np.take(currentDataP, dist13_21_pattern)  
            dist22_30_psdVals = np.take(currentDataP, dist22_30_pattern) 
            dist31_psdVals = np.take(currentDataP, dist31_plus_pattern) 
            
            #Remove nan entries 
            dist1_psdVals = dist1_psdVals[~np.isnan(dist1_psdVals)]
            dist2_psdVals = dist2_psdVals[~np.isnan(dist2_psdVals)]
            dist3_psdVals = dist3_psdVals[~np.isnan(dist3_psdVals)]
            dist4_psdVals = dist4_psdVals[~np.isnan(dist4_psdVals)]
            dist5_8_psdVals = dist5_8_psdVals[~np.isnan(dist5_8_psdVals)]
            dist9_12_psdVals = dist9_12_psdVals[~np.isnan(dist9_12_psdVals)]
            dist13_21_psdVals = dist13_21_psdVals[~np.isnan(dist13_21_psdVals)]
            dist22_30_psdVals = dist22_30_psdVals[~np.isnan(dist22_30_psdVals)]
            dist31_psdVals = dist31_psdVals[~np.isnan(dist31_psdVals)]
            
            #Prestimulus where available
            if pre == 1:
                dist1_psdVals_pre = np.take(currentDataRpre, dist1_pattern)
                #Remove nan entries
                dist1_psdVals_pre = dist1_psdVals_pre[~np.isnan(dist1_psdVals_pre)]
            
            #Remove Pattern Data that Doesn't Exist in Random 
            if currentStim == 'B1':
                dist1_psdVals = np.empty(len(dist1_psdVals))
                dist1_psdVals[:] = np.nan

            if currentStim == 'C2':
                dist3_psdVals = np.empty(len(dist3_psdVals))
                dist3_psdVals[:] = np.nan
                
            if currentStim == 'D1':
                dist2_psdVals = np.empty(len(dist2_psdVals))
                dist2_psdVals[:] = np.nan

            
            #Append sorted PSD means to meta holders
            if pre == 1: preStim.append(np.mean(dist1_psdVals_pre))
            else: preStim.append(np.nan)
            ms250.append(np.mean(dist1_psdVals))
            ms500.append(np.mean(dist2_psdVals))
            ms750.append(np.mean(dist3_psdVals))
            ms1000.append(np.mean(dist4_psdVals))
            s12.append(np.mean(dist5_8_psdVals))
            s23.append(np.mean(dist9_12_psdVals))
            s34.append(np.mean(dist13_21_psdVals))
            s45.append(np.mean(dist22_30_psdVals))
            s5p.append(np.mean(dist31_psdVals))
            
            #Check for significance between groups with Kruskal-Wallis H test 
            #Corrected Kruskal-Wallis tests to match pattern and random data 
            if currentStim == 'B1':
                h_stat, p_val = stats.kruskal(dist2_psdVals, dist3_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)

            if currentStim == 'C2':
                h_stat, p_val = stats.kruskal(dist1_psdVals, dist2_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)

            if currentStim == 'D1':
                h_stat, p_val = stats.kruskal(dist1_psdVals, dist3_psdVals, \
                                          dist4_psdVals, dist5_8_psdVals, dist9_12_psdVals, \
                                          dist13_21_psdVals, dist22_30_psdVals, dist31_psdVals)   
            
            KWpval.append(p_val)
            
            #Compile data for plotting and Dunn posthoc-test
            datatoPlot = [dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, \
                                                  dist5_8_psdVals, dist9_12_psdVals, dist13_21_psdVals, \
                                                  dist22_30_psdVals, dist31_psdVals]
            
            #Complete Dunn tests and append 
            if p_val <0.05:
                dunn_vals = sp.posthoc_dunn(datatoPlot)
                dunnPosthoc.append(np.hstack((np.asarray(dunn_vals).flatten(), np.zeros(64))))
            else: dunnPosthoc.append(np.zeros(145)) #81 for pattern, 64 for random   
                
            #Clear PSD Value arrays 
            dist1_psdVals, dist2_psdVals, dist3_psdVals, dist4_psdVals, dist5_8_psdVals,\
            dist9_12_psdVals, dist13_21_psdVals,dist22_30_psdVals, dist31_psdVals = None, None, None, None, \
            None, None, None, None, None     
                
######################################################################################################
#Population Analyses
######################################################################################################
#PREPARE DATA     
######Save meta data to file 
    animal = np.asarray(animal)
    layer = np.asarray(layer)
    paradigm = np.asarray(paradigm)
    stim = np.asarray(stim)
    freq = np.asarray(freq)
    KWpval = np.asarray(KWpval)
    dunnPosthoc = np.asarray(dunnPosthoc)
        
    datatofile = np.array([animal, layer, paradigm, stim, freq, KWpval])
    datatofile = datatofile.T 
    datatofile = np.hstack((datatofile, dunnPosthoc))
    
    dunn_arrays = np.hstack((np.arange(81), np.arange(64)))
    headers = np.hstack((['Animal', 'Layer', 'Paradigm','Whisker','FreqHz','KWpval'], dunn_arrays))
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/DistancefromProbe/MetaResultsTest_'\
                                    +name+'.csv', header = headers)           
    
######Data to arrays 
    #Mean PSD values     
    preStim = np.asarray(preStim)
    ms250 = np.asarray(ms250)
    ms500 = np.asarray(ms500)
    ms750 = np.asarray(ms750)
    ms1000 = np.asarray(ms1000)
    s12 = np.asarray(s12)
    s23 = np.asarray(s23)
    s34 = np.asarray(s34)
    s45 = np.asarray(s45)
    s5p = np.asarray(s5p)
    
#####Create DataFrame for easier data queries 
    compLayer = laminar_labelTolayer(layer)
    df = pd.DataFrame({'animal': animal, 'layer': layer, 'compLayer': compLayer, 'paradigm': paradigm, \
                       'whisker': stim, 'freq': freq, 'KWpval':KWpval, \
                      'preStimMean': preStim, 'ms250Mean': ms250, 'ms500Mean': ms500, 'ms750Mean': ms750, \
                      'ms1000Mean': ms1000, 's12Mean': s12, 's23Mean': s23, 's34Mean': s34, 's45Mean': s45, \
                      's5pMean': s5p}) 
    
#####Create holders for storing significance test results
    sig_Shank = []
    sig_Layer = []
    sig_Band = []
    sig_Protocol = []
    sig_Comp = []
    sig_pVal_Wx = [] 
    sig_n_one = []
    sig_n_two = []
    
######################################################################################################    
#META ANALYSES 
#####PERCENTAGE OF CHANNELS EXHIBITING MODULATION 

#####PATTERN 
    dfP = df[df['paradigm'] == 'P']
    
    #Breakdown into Alpha, beta, and Gamma Bands by layer 
    #SUPRAGRANULAR 
    SG_entries = dfP[dfP['compLayer'] == 'SG']
    SG_alpha = SG_entries[SG_entries['freq'].isin([8,12])] 
    SG_beta = SG_entries[SG_entries['freq'].isin([16,20,24,28])] 
    SG_gamma = SG_entries[SG_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    SG_alpha_mod = np.shape(SG_alpha[SG_alpha['KWpval'] < 0.05])[0] / np.shape(SG_alpha)[0]
    SG_beta_mod = np.shape(SG_beta[SG_beta['KWpval'] < 0.05])[0] / np.shape(SG_beta)[0]
    SG_gamma_mod = np.shape(SG_gamma[SG_gamma['KWpval'] < 0.05])[0] / np.shape(SG_gamma)[0]
    
    #For later-pre comparison analysis
    SG_beta_20 = SG_entries[SG_entries['freq'].isin([20])] 
    SG_gamma_4060 = SG_entries[SG_entries['freq'].isin([40,60])] 
    
    #GRANULAR 
    G_entries = dfP[dfP['compLayer'] == 'G']
    G_alpha = G_entries[G_entries['freq'].isin([8,12])] 
    G_beta = G_entries[G_entries['freq'].isin([16,20,24,28])] 
    G_gamma = G_entries[G_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    G_alpha_mod = np.shape(G_alpha[G_alpha['KWpval'] < 0.05])[0] / np.shape(G_alpha)[0]
    G_beta_mod = np.shape(G_beta[G_beta['KWpval'] < 0.05])[0] / np.shape(G_beta)[0]
    G_gamma_mod = np.shape(G_gamma[G_gamma['KWpval'] < 0.05])[0] / np.shape(G_gamma)[0]
    
    #For later-pre comparison analysis
    G_beta_20 = G_entries[G_entries['freq'].isin([20])] 
    G_gamma_4060 = G_entries[G_entries['freq'].isin([40,60])] 
    
    #UPPER INFRAGRANULAR 
    IGU_entries = dfP[dfP['compLayer'] == 'IGU']
    IGU_alpha = IGU_entries[IGU_entries['freq'].isin([8,12])] 
    IGU_beta = IGU_entries[IGU_entries['freq'].isin([16,20,24,28])] 
    IGU_gamma = IGU_entries[IGU_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    IGU_alpha_mod = np.shape(IGU_alpha[IGU_alpha['KWpval'] < 0.05])[0] / np.shape(IGU_alpha)[0]
    IGU_beta_mod = np.shape(IGU_beta[IGU_beta['KWpval'] < 0.05])[0] / np.shape(IGU_beta)[0]
    IGU_gamma_mod = np.shape(IGU_gamma[IGU_gamma['KWpval'] < 0.05])[0] / np.shape(IGU_gamma)[0]
    
    #LOWER INFRAGRANULAR
    IGL_entries = dfP[dfP['compLayer'] == 'IGL']
    IGL_alpha = IGL_entries[IGL_entries['freq'].isin([8,12])] 
    IGL_beta = IGL_entries[IGL_entries['freq'].isin([16,20,24,28])] 
    IGL_gamma = IGL_entries[IGL_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    IGL_alpha_mod = np.shape(IGL_alpha[IGL_alpha['KWpval'] < 0.05])[0] / np.shape(IGL_alpha)[0]
    IGL_beta_mod = np.shape(IGL_beta[IGL_beta['KWpval'] < 0.05])[0] / np.shape(IGL_beta)[0]
    IGL_gamma_mod = np.shape(IGL_gamma[IGL_gamma['KWpval'] < 0.05])[0] / np.shape(IGL_gamma)[0]
    
    #Sorted modulation rates
    alphaData = ([SG_alpha_mod, G_alpha_mod, IGU_alpha_mod, IGL_alpha_mod])
    betaData = ([SG_beta_mod, G_beta_mod,  IGU_beta_mod, IGL_beta_mod])
    gammaData = ([SG_gamma_mod, G_gamma_mod, IGU_gamma_mod, IGL_gamma_mod])
    
    ##PLOT OUTCOME 
    plot_laminar_profile_modulation_(alphaData, betaData, gammaData, path +\
                            '/Analysis/DistancefromProbe/PSDpercentModulation_Pattern_' + name + '.png')   
    
#####RANDOM
    dfR = df[df['paradigm'] == 'R']
    
    #Breakdown into Alpha, beta, and Gamma Bands by layer 
    #SUPRAGRANULAR 
    SG_entries_R = dfR[dfR['compLayer'] == 'SG']
    SG_alpha_R = SG_entries_R[SG_entries_R['freq'].isin([8,12])] 
    SG_beta_R = SG_entries_R[SG_entries_R['freq'].isin([16,20,24,28])] 
    SG_gamma_R = SG_entries_R[SG_entries_R['freq'].isin([32,36,40,44,48,52,56,60])] 
    SG_alpha_R_mod = np.shape(SG_alpha_R[SG_alpha_R['KWpval'] < 0.05])[0] / np.shape(SG_alpha_R)[0]
    SG_beta_R_mod = np.shape(SG_beta_R[SG_beta_R['KWpval'] < 0.05])[0] / np.shape(SG_beta_R)[0]
    SG_gamma_R_mod = np.shape(SG_gamma_R[SG_gamma_R['KWpval'] < 0.05])[0] / np.shape(SG_gamma_R)[0]
    
    #For later-pre comparison analysis
    SG_beta_R_20 = SG_entries_R[SG_entries_R['freq'].isin([20])] 
    SG_gamma_R_4060 = SG_entries_R[SG_entries_R['freq'].isin([40,60])] 
    
    #GRANULAR
    G_entries_R = dfR[dfR['compLayer'] == 'G']
    G_alpha_R = G_entries_R[G_entries_R['freq'].isin([8,12])] 
    G_beta_R = G_entries_R[G_entries_R['freq'].isin([16,20,24,28])] 
    G_gamma_R = G_entries_R[G_entries_R['freq'].isin([32,36,40,44,48,52,56,60])] 
    G_alpha_R_mod = np.shape(G_alpha_R[G_alpha_R['KWpval'] < 0.05])[0] / np.shape(G_alpha_R)[0]
    G_beta_R_mod = np.shape(G_beta_R[G_beta_R['KWpval'] < 0.05])[0] / np.shape(G_beta_R)[0]
    G_gamma_R_mod = np.shape(G_gamma_R[G_gamma_R['KWpval'] < 0.05])[0] / np.shape(G_gamma_R)[0]
    
    #For later-pre comparison analysis
    G_beta_R_20 = G_entries_R[G_entries_R['freq'].isin([20])] 
    G_gamma_R_4060 = G_entries_R[G_entries_R['freq'].isin([40,60])] 
    
    #UPPER INFRAGRANULAR 
    IGU_entries_R = dfR[dfR['compLayer'] == 'IGU']
    IGU_alpha_R = IGU_entries_R[IGU_entries_R['freq'].isin([8,12])] 
    IGU_beta_R = IGU_entries_R[IGU_entries_R['freq'].isin([16,20,24,28])] 
    IGU_gamma_R = IGU_entries_R[IGU_entries_R['freq'].isin([32,36,40,44,48,52,56,60])] 
    IGU_alpha_R_mod = np.shape(IGU_alpha_R[IGU_alpha_R['KWpval'] < 0.05])[0] / np.shape(IGU_alpha_R)[0]
    IGU_beta_R_mod = np.shape(IGU_beta_R[IGU_beta_R['KWpval'] < 0.05])[0] / np.shape(IGU_beta_R)[0]
    IGU_gamma_R_mod = np.shape(IGU_gamma_R[IGU_gamma_R['KWpval'] < 0.05])[0] / np.shape(IGU_gamma_R)[0]
    
    #LOWER INFRAGRANULAR 
    IGL_entries_R = dfR[dfR['compLayer'] == 'IGL']
    IGL_alpha_R = IGL_entries_R[IGL_entries_R['freq'].isin([8,12])] 
    IGL_beta_R = IGL_entries_R[IGL_entries_R['freq'].isin([16,20,24,28])] 
    IGL_gamma_R = IGL_entries_R[IGL_entries_R['freq'].isin([32,36,40,44,48,52,56,60])] 
    IGL_alpha_R_mod = np.shape(IGL_alpha_R[IGL_alpha_R['KWpval'] < 0.05])[0] / np.shape(IGL_alpha_R)[0]
    IGL_beta_R_mod = np.shape(IGL_beta_R[IGL_beta_R['KWpval'] < 0.05])[0] / np.shape(IGL_beta_R)[0]
    IGL_gamma_R_mod = np.shape(IGL_gamma_R[IGL_gamma_R['KWpval'] < 0.05])[0] / np.shape(IGL_gamma_R)[0]
    
    #Sorted modulation rates
    alphaData_R = ([SG_alpha_R_mod, G_alpha_R_mod, IGU_alpha_R_mod, IGL_alpha_R_mod])
    betaData_R = ([SG_beta_R_mod, G_beta_R_mod,  IGU_beta_R_mod, IGL_beta_R_mod])
    gammaData_R = ([SG_gamma_R_mod, G_gamma_R_mod, IGU_gamma_R_mod, IGL_gamma_R_mod])
    
    #Plot Outcome 
    plot_laminar_profile_modulation_(alphaData_R,betaData_R , gammaData_R, path +\
                            '/Analysis/DistancefromProbe/PSDpercentModulation_Random_' + name + '.png')   
    
#####PATTERN Corrected 
    dfP_Corr = df[df['paradigm'] == 'P_Corr']
    
    #Breakdown into Alpha, beta, and Gamma Bands by layer 
    #SUPRAGRANULAR 
    Corr_SG_entries = dfP_Corr[dfP_Corr['compLayer'] == 'SG']
    Corr_SG_alpha = Corr_SG_entries[Corr_SG_entries['freq'].isin([8,12])] 
    Corr_SG_beta = Corr_SG_entries[Corr_SG_entries['freq'].isin([16,20,24,28])] 
    Corr_SG_gamma = Corr_SG_entries[Corr_SG_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    Corr_SG_alpha_mod = np.shape(Corr_SG_alpha[Corr_SG_alpha['KWpval'] < 0.05])[0] / np.shape(Corr_SG_alpha)[0]
    Corr_SG_beta_mod = np.shape(Corr_SG_beta[Corr_SG_beta['KWpval'] < 0.05])[0] / np.shape(Corr_SG_beta)[0]
    Corr_SG_gamma_mod = np.shape(Corr_SG_gamma[Corr_SG_gamma['KWpval'] < 0.05])[0] / np.shape(Corr_SG_gamma)[0]
    
    #For later-pre comparison analysis
    Corr_SG_beta_20 = Corr_SG_entries[Corr_SG_entries['freq'].isin([20])] 
    Corr_SG_gamma_4060 = Corr_SG_entries[Corr_SG_entries['freq'].isin([40,60])] 
    
    #GRANULAR 
    Corr_G_entries = dfP_Corr[dfP_Corr['compLayer'] == 'G']
    Corr_G_alpha = Corr_G_entries[Corr_G_entries['freq'].isin([8,12])] 
    Corr_G_beta = Corr_G_entries[Corr_G_entries['freq'].isin([16,20,24,28])] 
    Corr_G_gamma = Corr_G_entries[Corr_G_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    Corr_G_alpha_mod = np.shape(Corr_G_alpha[Corr_G_alpha['KWpval'] < 0.05])[0] / np.shape(Corr_G_alpha)[0]
    Corr_G_beta_mod = np.shape(Corr_G_beta[Corr_G_beta['KWpval'] < 0.05])[0] / np.shape(Corr_G_beta)[0]
    Corr_G_gamma_mod = np.shape(Corr_G_gamma[Corr_G_gamma['KWpval'] < 0.05])[0] / np.shape(Corr_G_gamma)[0]
    
    #For later-pre comparison analysis
    Corr_G_beta_20 = Corr_G_entries[Corr_G_entries['freq'].isin([20])] 
    Corr_G_gamma_4060 = Corr_G_entries[Corr_G_entries['freq'].isin([40,60])] 
    
    #UPPER INFRAGRANULAR 
    Corr_IGU_entries = dfP_Corr[dfP_Corr['compLayer'] == 'IGU']
    Corr_IGU_alpha = Corr_IGU_entries[Corr_IGU_entries['freq'].isin([8,12])] 
    Corr_IGU_beta = Corr_IGU_entries[Corr_IGU_entries['freq'].isin([16,20,24,28])] 
    Corr_IGU_gamma = Corr_IGU_entries[Corr_IGU_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    Corr_IGU_alpha_mod = np.shape(Corr_IGU_alpha[Corr_IGU_alpha['KWpval'] <0.05])[0]/np.shape(Corr_IGU_alpha)[0]
    Corr_IGU_beta_mod = np.shape(Corr_IGU_beta[Corr_IGU_beta['KWpval'] < 0.05])[0] / np.shape(Corr_IGU_beta)[0]
    Corr_IGU_gamma_mod = np.shape(Corr_IGU_gamma[Corr_IGU_gamma['KWpval'] < 0.05])[0] / np.shape(Corr_IGU_gamma)[0]
    
    #LOWER INFRAGRANULAR
    Corr_IGL_entries = dfP_Corr[dfP_Corr['compLayer'] == 'IGL']
    Corr_IGL_alpha = Corr_IGL_entries[Corr_IGL_entries['freq'].isin([8,12])] 
    Corr_IGL_beta = Corr_IGL_entries[Corr_IGL_entries['freq'].isin([16,20,24,28])] 
    Corr_IGL_gamma = Corr_IGL_entries[Corr_IGL_entries['freq'].isin([32,36,40,44,48,52,56,60])] 
    Corr_IGL_alpha_mod = np.shape(Corr_IGL_alpha[Corr_IGL_alpha['KWpval']<0.05])[0]/np.shape(Corr_IGL_alpha)[0]
    Corr_IGL_beta_mod = np.shape(Corr_IGL_beta[Corr_IGL_beta['KWpval'] < 0.05])[0] / np.shape(Corr_IGL_beta)[0]
    Corr_IGL_gamma_mod = np.shape(Corr_IGL_gamma[Corr_IGL_gamma['KWpval'] <0.05])[0]/np.shape(Corr_IGL_gamma)[0]
    
    #Sorted modulation rates
    Corr_alphaData = ([Corr_SG_alpha_mod, Corr_G_alpha_mod, Corr_IGU_alpha_mod, Corr_IGL_alpha_mod])
    Corr_betaData = ([Corr_SG_beta_mod, Corr_G_beta_mod,  Corr_IGU_beta_mod, Corr_IGL_beta_mod])
    Corr_gammaData = ([Corr_SG_gamma_mod, Corr_G_gamma_mod, Corr_IGU_gamma_mod, Corr_IGL_gamma_mod])
    
    ##PLOT OUTCOME 
    plot_laminar_profile_modulation_(Corr_alphaData, Corr_betaData , Corr_gammaData, path+\
                            '/Analysis/DistancefromProbe/PSDpercentModulation_PatternCorrected_' + name + '.png')     

######################################################################################################    
#Plot Comparison
    plot_laminar_profile_modulation_overlay(alphaData, betaData, gammaData, \
                                            Corr_alphaData, Corr_betaData, Corr_gammaData, \
                                            alphaData_R, betaData_R, gammaData_R,\
        path +'/Analysis/DistancefromProbe/PSDpercentModulation_AllComparisons_' + name + '.png')
    
######################################################################################################    
#META ANALYSES 
#####Batched Mean Power [pre, 0-750ms, 1+s] 

##Holders for Comparisons between Pattern and Random
    comp_layer = []
    comp_band = []
    comparison = []
    n_one = []
    n_two = [] 
    comp_pattern_mean1 = []
    comp_pattern_mean2 = []
    comp_pattern_diff =[]
    comp_random_mean1 = []
    comp_random_mean2 = []
    comp_random_diff = []
    comp_pattern_corr_mean1 = []
    comp_pattern_corr_mean2 = []
    comp_pattern_corr_diff =[]
    comp_Wx_pVal = []
    comp_Wx_pVal_Corr = []

####PATTERN
  ###SUPRAGRANULAR 
   ##Alpha 
    SGalpha750 = np.array([SG_alpha['ms250Mean'], SG_alpha['ms500Mean'], SG_alpha['ms750Mean']]).flatten()
    SGalpha1p = np.array([SG_alpha['ms1000Mean'], SG_alpha['s12Mean'], SG_alpha['s23Mean'], \
                          SG_alpha['s34Mean'], SG_alpha['s45Mean'],SG_alpha['s5pMean']]).flatten()
    SGalphaData = [SGalpha750, SGalpha1p]
    tstat_W, p_val_W = stats.ranksums(SGalpha750, SGalpha1p)
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Alpha')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGalpha750))
    sig_n_two.append(len(SGalpha1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('SG')
    comp_band.append('Alpha')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([SG_alpha['ms250Mean'], SG_alpha['ms500Mean'], SG_alpha['ms750Mean']]), axis=0) 
    late = np.mean(np.array([SG_alpha['ms1000Mean'], SG_alpha['s12Mean'], SG_alpha['s23Mean'], \
                          SG_alpha['s34Mean'], SG_alpha['s45Mean'],SG_alpha['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_SG_alpha = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_SG_alpha))
    
    
   ##Beta
    SGbeta750 = np.array([SG_beta['ms250Mean'], SG_beta['ms500Mean'], SG_beta['ms750Mean']]).flatten()
    SGbeta1p = np.array([SG_beta['ms1000Mean'], SG_beta['s12Mean'], SG_beta['s23Mean'], \
                          SG_beta['s34Mean'], SG_beta['s45Mean'],SG_beta['s5pMean']]).flatten()
    SGbetaData = [SGbeta750, SGbeta1p]
    tstat_W, p_val_W = stats.ranksums(SGbeta750, SGbeta1p)

    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbeta750))
    sig_n_two.append(len(SGbeta1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('SG')
    comp_band.append('Beta')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([SG_beta['ms250Mean'], SG_beta['ms500Mean'], SG_beta['ms750Mean']]), axis=0) 
    late = np.mean(np.array([SG_beta['ms1000Mean'], SG_beta['s12Mean'], SG_beta['s23Mean'], \
                          SG_beta['s34Mean'], SG_beta['s45Mean'],SG_beta['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_SG_beta = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_SG_beta))
                    
    #20Hz for Prestimulus comparison 
    SGbetapre = np.array(SG_beta['preStimMean'])
    SGbetapre = SGbetapre[~np.isnan(SGbetapre)]  #Select only 20Hz entries (all others are np.nan)
    SGbeta750_20 = np.array([SG_beta_20['ms250Mean'],SG_beta_20['ms500Mean'],SG_beta_20['ms750Mean']]).flatten()
    SGbeta1p_20 = np.array([SG_beta_20['ms1000Mean'], SG_beta_20['s12Mean'], SG_beta_20['s23Mean'], \
                          SG_beta_20['s34Mean'], SG_beta_20['s45Mean'],SG_beta_20['s5pMean']]).flatten()
    SGbetaDataPre = [SGbetapre, SGbeta750_20, SGbeta1p_20]
    
    tstat_W, p_val_W = stats.ranksums(SGbetapre, SGbeta750_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('pre-750') 
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbetapre))
    sig_n_two.append(len(SGbeta750_20))
    
    tstat_W, p_val_W = stats.ranksums(SGbetapre, SGbeta1p_20)      
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('pre-1p')  
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbetapre))
    sig_n_two.append(len(SGbeta1p_20))
    
    tstat_W, p_val_W = stats.ranksums(SGbeta750_20, SGbeta1p_20) 
        
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbeta750_20))
    sig_n_two.append(len(SGbeta1p_20))
    
   ##Gamma 
    SGgamma750 = np.array([SG_gamma['ms250Mean'], SG_gamma['ms500Mean'], SG_gamma['ms750Mean']]).flatten()
    SGgamma1p = np.array([SG_gamma['ms1000Mean'], SG_gamma['s12Mean'], SG_gamma['s23Mean'], \
                          SG_gamma['s34Mean'], SG_gamma['s45Mean'],SG_gamma['s5pMean']]).flatten()
    SGgammaData = [SGgamma750, SGgamma1p]
    tstat_W, p_val_W = stats.ranksums(SGgamma750, SGgamma1p)
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgamma750))
    sig_n_two.append(len(SGgamma1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('SG')
    comp_band.append('Gamma')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([SG_gamma['ms250Mean'], SG_gamma['ms500Mean'], SG_gamma['ms750Mean']]), axis=0) 
    late = np.mean(np.array([SG_gamma['ms1000Mean'], SG_gamma['s12Mean'], SG_gamma['s23Mean'], \
                          SG_gamma['s34Mean'], SG_gamma['s45Mean'],SG_gamma['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_SG_gamma = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_SG_gamma))
    
    #40 & 60Hz for Prestimulus comparison 
    SGgammapre = np.array(SG_gamma['preStimMean'])
    SGgammapre = SGgammapre[~np.isnan(SGgammapre)] #Select only 40 & 60Hz entries (all others are np.nan)
    SGgamma750_4060 = np.array([SG_gamma_4060['ms250Mean'],SG_gamma_4060['ms500Mean'],\
                               SG_gamma_4060['ms750Mean']]).flatten()
    SGgamma1p_4060 = np.array([SG_gamma_4060['ms1000Mean'], SG_gamma_4060['s12Mean'], \
                               SG_gamma_4060['s23Mean'], SG_gamma_4060['s34Mean'], \
                               SG_gamma_4060['s45Mean'],SG_gamma_4060['s5pMean']]).flatten()
    SGgammaDataPre = np.array([SGgammapre, SGgamma750_4060, SGgamma1p_4060])

    tstat_W, p_val_W = stats.ranksums(SGgammapre, SGgamma750_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('pre-750')     
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgammapre))
    sig_n_two.append(len(SGgamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(SGgammapre, SGgamma750_4060)     
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgammapre))
    sig_n_two.append(len(SGgamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(SGgamma750_4060, SGgamma1p_4060) 
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgamma750_4060))
    sig_n_two.append(len(SGgamma1p_4060))
    
  ###GRANULAR 
   ##Alpha   
    Galpha750 = np.array([G_alpha['ms250Mean'], G_alpha['ms500Mean'], G_alpha['ms750Mean']]).flatten()
    Galpha1p = np.array([G_alpha['ms1000Mean'], G_alpha['s12Mean'], G_alpha['s23Mean'], \
                          G_alpha['s34Mean'], G_alpha['s45Mean'],G_alpha['s5pMean']]).flatten()
    GalphaData = [Galpha750, Galpha1p]
    tstat_W, p_val_W = stats.ranksums(Galpha750, Galpha1p)
 
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Alpha')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Galpha750))
    sig_n_two.append(len(Galpha1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('G')
    comp_band.append('Alpha')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([G_alpha['ms250Mean'], G_alpha['ms500Mean'], G_alpha['ms750Mean']]), axis=0) 
    late = np.mean(np.array([G_alpha['ms1000Mean'], G_alpha['s12Mean'], G_alpha['s23Mean'], \
                          G_alpha['s34Mean'], G_alpha['s45Mean'],G_alpha['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_G_alpha = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_G_alpha))

   ##Beta   
    Gbeta750 = np.array([G_beta['ms250Mean'], G_beta['ms500Mean'], G_beta['ms750Mean']]).flatten()
    Gbeta1p = np.array([G_beta['ms1000Mean'], G_beta['s12Mean'], G_beta['s23Mean'], \
                          G_beta['s34Mean'], G_beta['s45Mean'],G_beta['s5pMean']]).flatten()
    GbetaData = [Gbeta750, Gbeta1p]
    tstat_W, p_val_W = stats.ranksums(Gbeta750, Gbeta1p)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbeta750))
    sig_n_two.append(len(Gbeta1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('G')
    comp_band.append('Beta')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([G_beta['ms250Mean'], G_beta['ms500Mean'], G_beta['ms750Mean']]), axis=0) 
    late = np.mean(np.array([G_beta['ms1000Mean'], G_beta['s12Mean'], G_beta['s23Mean'], \
                          G_beta['s34Mean'], G_beta['s45Mean'],G_beta['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_G_beta = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_G_beta))
    
    #20Hz for Prestimulus comparison
    Gbetapre = np.array(G_beta['preStimMean'])
    Gbetapre = Gbetapre[~np.isnan(Gbetapre)]  #Select only 20Hz entries (all others are np.nan)
    Gbeta750_20 = np.array([G_beta_20['ms250Mean'],G_beta_20['ms500Mean'],G_beta_20['ms750Mean']]).flatten()
    Gbeta1p_20 = np.array([G_beta_20['ms1000Mean'], G_beta_20['s12Mean'], G_beta_20['s23Mean'], \
                          G_beta_20['s34Mean'], G_beta_20['s45Mean'],G_beta_20['s5pMean']]).flatten()
    GbetaDataPre = np.array([Gbetapre, Gbeta750_20, Gbeta1p_20])
    
    tstat_W, p_val_W = stats.ranksums(Gbetapre, Gbeta750_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('pre-750') 
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbetapre))
    sig_n_two.append(len(Gbeta750_20))
    
    tstat_W, p_val_W = stats.ranksums(Gbetapre, Gbeta1p_20)     
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbetapre))
    sig_n_two.append(len(Gbeta1p_20))
    
    tstat_W, p_val_W = stats.ranksums(Gbeta750_20, Gbeta1p_20) 
        
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbeta750_20))
    sig_n_two.append(len(Gbeta1p_20))

   ##Gamma 
    Ggamma750 = np.array([G_gamma['ms250Mean'], G_gamma['ms500Mean'], G_gamma['ms750Mean']]).flatten()
    Ggamma1p = np.array([G_gamma['ms1000Mean'], G_gamma['s12Mean'], G_gamma['s23Mean'], \
                          G_gamma['s34Mean'], G_gamma['s45Mean'],G_gamma['s5pMean']]).flatten()
    GgammaData = [Ggamma750, Ggamma1p]
    tstat_W, p_val_W = stats.ranksums(Ggamma750, Ggamma1p)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggamma750))
    sig_n_two.append(len(Ggamma1p))
    
    #Comparison between Pattern and Random
    comp_layer.append('G')
    comp_band.append('Gamma')
    comparison.append('0-750AND1s+')
    early = np.mean(np.array([G_gamma['ms250Mean'], G_gamma['ms500Mean'], G_gamma['ms750Mean']]), axis=0) 
    late = np.mean(np.array([G_gamma['ms1000Mean'], G_gamma['s12Mean'], G_gamma['s23Mean'], \
                          G_gamma['s34Mean'], G_gamma['s45Mean'],G_gamma['s5pMean']]), axis=0)
    mean_diff = early - late 
    P_G_gamma = mean_diff
    comp_pattern_mean1.append(np.mean(early))
    comp_pattern_mean2.append(np.mean(late))
    comp_pattern_diff.append(np.mean(P_G_gamma))
    
    #40 & 60Hz for Prestimulus comparison
    Ggammapre = np.array(G_gamma['preStimMean'])
    Ggammapre = Ggammapre[~np.isnan(Ggammapre)]  #Select only 40 & 60Hz entries (all others are np.nan)
    Ggamma750_4060 = np.array([G_gamma_4060['ms250Mean'],G_gamma_4060['ms500Mean'],\
                               G_gamma_4060['ms750Mean']]).flatten()
    Ggamma1p_4060 = np.array([G_gamma_4060['ms1000Mean'],G_gamma_4060['s12Mean'], G_gamma_4060['s23Mean'], \
                          G_gamma_4060['s34Mean'], G_gamma_4060['s45Mean'],G_gamma_4060['s5pMean']]).flatten()
    GgammaDataPre = np.array([Ggammapre, Ggamma750_4060, Ggamma1p_4060])

    tstat_W, p_val_W = stats.ranksums(Ggammapre, Ggamma750_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('pre-750')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggammapre))
    sig_n_two.append(len(Ggamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(Ggammapre, Ggamma1p_4060)
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggammapre))
    sig_n_two.append(len(Ggamma1p_4060))
    
    tstat_W, p_val_W = stats.ranksums(Ggamma750_4060, Ggamma1p_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggamma750_4060))
    sig_n_two.append(len(Ggamma1p_4060))
  
 ###RANDOM
  ###SUPRAGRANULAR 
   ##Alpha  
    SGalpha750_R = np.array([SG_alpha_R['ms250Mean'], SG_alpha_R['ms500Mean'], \
                             SG_alpha_R['ms750Mean']]).flatten()
    SGalpha750_R = SGalpha750_R[~np.isnan(SGalpha750_R)] #Remove non-entries that are stored as np.nans
    SGalpha1p_R = np.array([SG_alpha_R['ms1000Mean'], SG_alpha_R['s12Mean'], SG_alpha_R['s23Mean'], \
                          SG_alpha_R['s34Mean'], SG_alpha_R['s45Mean'],SG_alpha_R['s5pMean']]).flatten()
    SGalphaData_R = [SGalpha750_R, SGalpha1p_R]
    tstat_W, p_val_W = stats.ranksums(SGalpha750_R, SGalpha1p_R)

    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Alpha')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGalpha750_R))
    sig_n_two.append(len(SGalpha1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([SG_alpha_R['ms250Mean'],SG_alpha_R['ms500Mean'],SG_alpha_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([SG_alpha_R['ms1000Mean'], SG_alpha_R['s12Mean'], SG_alpha_R['s23Mean'], \
                          SG_alpha_R['s34Mean'], SG_alpha_R['s45Mean'],SG_alpha_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_SG_alpha = mean_diff
    P_SG_alpha = np.asarray(P_SG_alpha).flatten()
    R_SG_alpha = np.asarray(R_SG_alpha).flatten()
    stat, pVal = stats.wilcoxon(P_SG_alpha, R_SG_alpha)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_SG_alpha))
    comp_Wx_pVal.append(pVal)

   ##Beta
    SGbeta750_R = np.array([SG_beta_R['ms250Mean'], SG_beta_R['ms500Mean'], SG_beta_R['ms750Mean']]).flatten()
    SGbeta750_R = SGbeta750_R[~np.isnan(SGbeta750_R)]
    SGbeta1p_R = np.array([SG_beta_R['ms1000Mean'], SG_beta_R['s12Mean'], SG_beta_R['s23Mean'], \
                          SG_beta_R['s34Mean'], SG_beta_R['s45Mean'],SG_beta_R['s5pMean']]).flatten()
    SGbetaData_R = [SGbeta750_R, SGbeta1p_R]
    tstat_W, p_val_W = stats.ranksums(SGbeta750_R, SGbeta1p_R)

    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbeta750_R))
    sig_n_two.append(len(SGbeta1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([SG_beta_R['ms250Mean'],SG_beta_R['ms500Mean'],SG_beta_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([SG_beta_R['ms1000Mean'], SG_beta_R['s12Mean'], SG_beta_R['s23Mean'], \
                          SG_beta_R['s34Mean'], SG_beta_R['s45Mean'],SG_beta_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_SG_beta = mean_diff
    P_SG_beta = np.asarray(P_SG_beta).flatten()
    R_SG_beta = np.asarray(R_SG_beta).flatten()
    stat, pVal = stats.wilcoxon(P_SG_beta, R_SG_beta)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_SG_beta))
    comp_Wx_pVal.append(pVal)
    
    #20Hz for Prestimulus comparison 
    SGbetapre_R = np.array(SG_beta_R['preStimMean'])
    SGbetapre_R = SGbetapre_R[~np.isnan(SGbetapre_R)]  #Select only 20Hz entries (all others are np.nan)
    SGbeta750_R_20 = np.array([SG_beta_R_20['ms250Mean'],SG_beta_R_20['ms500Mean'],\
                               SG_beta_R_20['ms750Mean']]).flatten()
    SGbeta750_R_20 = SGbeta750_R_20[~np.isnan(SGbeta750_R_20)] #Remove non-entries that are stored as np.nans
    SGbeta1p_R_20 = np.array([SG_beta_R_20['ms1000Mean'], SG_beta_R_20['s12Mean'], SG_beta_R_20['s23Mean'], \
                          SG_beta_R_20['s34Mean'], SG_beta_R_20['s45Mean'],SG_beta_R_20['s5pMean']]).flatten()
    SGbetaDataPre_R = [SGbetapre_R, SGbeta750_R_20, SGbeta1p_R_20]

    tstat_W, p_val_W = stats.ranksums(SGbetapre_R, SGbeta750_R_20)
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('pre-750')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbetapre_R))
    sig_n_two.append(len(SGbeta750_R_20))
    
    tstat_W, p_val_W = stats.ranksums(SGbetapre_R, SGbeta1p_R_20) 
         
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbetapre_R))
    sig_n_two.append(len(SGbeta1p_R_20))
    
    tstat_W, p_val_W = stats.ranksums(SGbeta750_R_20, SGbeta1p_R_20) 
        
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGbeta750_R_20))
    sig_n_two.append(len(SGbeta1p_R_20))
    
   ##Gamma 
    SGgamma750_R = np.array([SG_gamma_R['ms250Mean'], SG_gamma_R['ms500Mean'], \
                             SG_gamma_R['ms750Mean']]).flatten()
    SGgamma750_R = SGgamma750_R[~np.isnan(SGgamma750_R)]  #Remove non-entries that are stored as np.nans
    SGgamma1p_R = np.array([SG_gamma_R['ms1000Mean'], SG_gamma_R['s12Mean'], SG_gamma_R['s23Mean'], \
                          SG_gamma_R['s34Mean'], SG_gamma_R['s45Mean'],SG_gamma_R['s5pMean']]).flatten()
    SGgammaData_R = [SGgamma750_R, SGgamma1p_R]
    tstat_W, p_val_W = stats.ranksums(SGgamma750_R, SGgamma1p_R)

    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgamma750_R))
    sig_n_two.append(len(SGgamma1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([SG_gamma_R['ms250Mean'],SG_gamma_R['ms500Mean'],SG_gamma_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([SG_gamma_R['ms1000Mean'], SG_gamma_R['s12Mean'], SG_gamma_R['s23Mean'], \
                          SG_gamma_R['s34Mean'], SG_gamma_R['s45Mean'],SG_gamma_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_SG_gamma = mean_diff
    P_SG_gamma = np.asarray(P_SG_gamma).flatten()
    R_SG_gamma = np.asarray(R_SG_gamma).flatten()
    stat, pVal = stats.wilcoxon(P_SG_gamma, R_SG_gamma)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_SG_gamma))
    comp_Wx_pVal.append(pVal)

    #40 & 60Hz for Prestimulus comparison
    SGgammapre_R = np.array(SG_gamma_R['preStimMean'])
    SGgammapre_R = SGgammapre_R[~np.isnan(SGgammapre_R)]  #Select only 40 & 60Hz entries (all others are np.nan)
    SGgamma750_R_4060 = np.array([SG_gamma_R_4060['ms250Mean'],SG_gamma_R_4060['ms500Mean'],\
                               SG_gamma_R_4060['ms750Mean']]).flatten()
    SGgamma750_R_4060 = SGgamma750_R_4060[~np.isnan(SGgamma750_R_4060)] #Remove np.nan entries 
    SGgamma1p_R_4060 = np.array([SG_gamma_R_4060['ms1000Mean'], SG_gamma_R_4060['s12Mean'], \
                                 SG_gamma_R_4060['s23Mean'], SG_gamma_R_4060['s34Mean'], \
                                 SG_gamma_R_4060['s45Mean'],SG_gamma_R_4060['s5pMean']]).flatten()
    SGgammaDataPre_R = [SGgammapre_R, SGgamma750_R_4060, SGgamma1p_R_4060]  
    
    tstat_W, p_val_W = stats.ranksums(SGgammapre_R, SGgamma750_R_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('pre-750')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgammapre_R))
    sig_n_two.append(len(SGgamma750_R_4060))
    
    tstat_W, p_val_W = stats.ranksums(SGgammapre_R, SGgamma1p_R_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgammapre_R))
    sig_n_two.append(len(SGgamma1p_R_4060))
    
    tstat_W, p_val_W = stats.ranksums(SGgamma750_R_4060, SGgamma1p_R_4060) 
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(SGgamma750_R_4060))
    sig_n_two.append(len(SGgamma1p_R_4060))
    
  ##GRANULAR
   ##Alpha  
    Galpha750_R = np.array([G_alpha_R['ms250Mean'], G_alpha_R['ms500Mean'], G_alpha_R['ms750Mean']]).flatten()
    Galpha750_R = Galpha750_R[~np.isnan(Galpha750_R)]
    Galpha1p_R = np.array([G_alpha_R['ms1000Mean'], G_alpha_R['s12Mean'], G_alpha_R['s23Mean'], \
                          G_alpha_R['s34Mean'], G_alpha_R['s45Mean'],G_alpha_R['s5pMean']]).flatten()
    GalphaData_R = [Galpha750_R, Galpha1p_R]
    tstat_W, p_val_W = stats.ranksums(Galpha750_R, Galpha1p_R)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Alpha')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Galpha750_R))
    sig_n_two.append(len(Galpha1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([G_alpha_R['ms250Mean'],G_alpha_R['ms500Mean'],G_alpha_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([G_alpha_R['ms1000Mean'], G_alpha_R['s12Mean'], G_alpha_R['s23Mean'], \
                          G_alpha_R['s34Mean'], G_alpha_R['s45Mean'],G_alpha_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_G_alpha = mean_diff
    P_G_alpha = np.asarray(P_G_alpha).flatten()
    R_G_alpha = np.asarray(R_G_alpha).flatten()
    stat, pVal = stats.wilcoxon(P_G_alpha, R_G_alpha)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_G_alpha))
    comp_Wx_pVal.append(pVal)

  ##Beta
    Gbeta750_R = np.array([G_beta_R['ms250Mean'], G_beta_R['ms500Mean'], G_beta_R['ms750Mean']]).flatten()
    Gbeta750_R = Gbeta750_R[~np.isnan(Gbeta750_R)]
    Gbeta1p_R = np.array([G_beta_R['ms1000Mean'], G_beta_R['s12Mean'], G_beta_R['s23Mean'], \
                          G_beta_R['s34Mean'], G_beta_R['s45Mean'],G_beta_R['s5pMean']]).flatten()
    GbetaData_R = [Gbeta750_R, Gbeta1p_R]
    tstat_W, p_val_W = stats.ranksums(Gbeta750_R, Gbeta1p_R)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbeta750_R))
    sig_n_two.append(len(Gbeta1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([G_beta_R['ms250Mean'],G_beta_R['ms500Mean'],G_beta_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([G_beta_R['ms1000Mean'], G_beta_R['s12Mean'], G_beta_R['s23Mean'], \
                          G_beta_R['s34Mean'], G_beta_R['s45Mean'],G_beta_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_G_beta = mean_diff
    P_G_beta = np.asarray(P_G_beta).flatten()
    R_G_beta = np.asarray(R_G_beta).flatten()
    stat, pVal = stats.wilcoxon(P_G_beta, R_G_beta)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_G_beta))
    comp_Wx_pVal.append(pVal)
    
    #20Hz for Prestimulus comparison 
    Gbetapre_R = np.array(G_beta_R['preStimMean'])
    Gbetapre_R = Gbetapre_R[~np.isnan(Gbetapre_R)]  #Select only 20Hz entries (all others are np.nan)
    Gbeta750_R_20 = np.array([G_beta_R_20['ms250Mean'],G_beta_R_20['ms500Mean'],\
                               G_beta_R_20['ms750Mean']]).flatten()
    Gbeta750_R_20 = Gbeta750_R_20[~np.isnan(Gbeta750_R_20)] #Remove non-entries that are stored as np.nans
    Gbeta1p_R_20 = np.array([G_beta_R_20['ms1000Mean'], G_beta_R_20['s12Mean'], G_beta_R_20['s23Mean'], \
                          G_beta_R_20['s34Mean'], G_beta_R_20['s45Mean'],G_beta_R_20['s5pMean']]).flatten()
    GbetaDataPre_R = [Gbetapre_R, Gbeta750_R_20, Gbeta1p_R_20]

    tstat_W, p_val_W = stats.ranksums(Gbetapre_R, Gbeta750_R_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('pre-750')   
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbetapre_R))
    sig_n_two.append(len(Gbeta750_R_20))

    tstat_W, p_val_W = stats.ranksums(Gbetapre_R, Gbeta1p_R_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbetapre_R))
    sig_n_two.append(len(Gbeta1p_R_20))
    
    tstat_W, p_val_W = stats.ranksums(Gbeta750_R_20, Gbeta1p_R_20)
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Gbeta750_R_20))
    sig_n_two.append(len(Gbeta1p_R_20))
    
  ##Gamma
    Ggamma750_R = np.array([G_gamma_R['ms250Mean'], G_gamma_R['ms500Mean'], G_gamma_R['ms750Mean']]).flatten()
    Ggamma750_R = Ggamma750_R[~np.isnan(Ggamma750_R)]
    Ggamma1p_R = np.array([G_gamma_R['ms1000Mean'], G_gamma_R['s12Mean'], G_gamma_R['s23Mean'], \
                          G_gamma_R['s34Mean'], G_gamma_R['s45Mean'],G_gamma_R['s5pMean']]).flatten()
    GgammaData_R = [Ggamma750_R, Ggamma1p_R]
    tstat_W, p_val_W = stats.ranksums(Ggamma750_R, Ggamma1p_R)

    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggamma750_R))
    sig_n_two.append(len(Ggamma1p_R))
    
    #Comparison of effects between Pattern and Random paradigms
    early = np.nanmean(np.array([G_gamma_R['ms250Mean'],G_gamma_R['ms500Mean'],G_gamma_R['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([G_gamma_R['ms1000Mean'], G_gamma_R['s12Mean'], G_gamma_R['s23Mean'], \
                          G_gamma_R['s34Mean'], G_gamma_R['s45Mean'],G_gamma_R['s5pMean']]), axis=0)
    mean_diff = early - late 
    R_G_gamma = mean_diff
    P_G_gamma = np.asarray(P_G_gamma).flatten()
    R_G_gamma = np.asarray(R_G_gamma).flatten()
    stat, pVal = stats.wilcoxon(P_G_gamma, R_G_gamma)
    comp_random_mean1.append(np.mean(early))
    comp_random_mean2.append(np.mean(late))
    comp_random_diff.append(np.mean(R_G_gamma))
    comp_Wx_pVal.append(pVal)
    
    #40 & 60Hz for Prestimulus comparison
    Ggammapre_R = np.array(G_gamma_R['preStimMean'])
    Ggammapre_R = Ggammapre_R[~np.isnan(Ggammapre_R)]  #Select only 40 & 60Hz entries (all others are np.nan)
    Ggamma750_R_4060 = np.array([G_gamma_R_4060['ms250Mean'],G_gamma_R_4060['ms500Mean'],\
                               G_gamma_R_4060['ms750Mean']]).flatten()
    Ggamma750_R_4060 = Ggamma750_R_4060[~np.isnan(Ggamma750_R_4060)] #Remove np.nan entries 
    Ggamma1p_R_4060 = np.array([G_gamma_R_4060['ms1000Mean'],G_gamma_R_4060['s12Mean'],\
                                G_gamma_R_4060['s23Mean'],  G_gamma_R_4060['s34Mean'], \
                                G_gamma_R_4060['s45Mean'],G_gamma_R_4060['s5pMean']]).flatten()
    GgammaDataPre_R = [Ggammapre_R, Ggamma750_R_4060, Ggamma1p_R_4060]

    tstat_W, p_val_W = stats.ranksums(Ggammapre_R, Ggamma750_R_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('pre-750')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggammapre_R))
    sig_n_two.append(len(Ggamma750_R_4060))
    
    tstat_W, p_val_W = stats.ranksums(Ggammapre_R, Ggamma1p_R_4060) 
        
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggammapre_R))
    sig_n_two.append(len(Ggamma1p_R_4060))
    
    tstat_W, p_val_W = stats.ranksums(Ggamma750_R_4060, Ggamma1p_R_4060) 
        
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('R')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Ggamma750_R_4060))
    sig_n_two.append(len(Ggamma1p_R_4060))
    
####PATTERN Corrected
  ###SUPRAGRANULAR 
   ##Alpha 
    Corr_SGalpha750 = np.array([Corr_SG_alpha['ms250Mean'], Corr_SG_alpha['ms500Mean'], \
                           Corr_SG_alpha['ms750Mean']]).flatten()
    Corr_SGalpha750 = Corr_SGalpha750[~np.isnan(Corr_SGalpha750)]
    Corr_SGalpha1p = np.array([Corr_SG_alpha['ms1000Mean'], Corr_SG_alpha['s12Mean'],Corr_SG_alpha['s23Mean'], \
                      Corr_SG_alpha['s34Mean'], Corr_SG_alpha['s45Mean'],Corr_SG_alpha['s5pMean']]).flatten()
    Corr_SGalphaData = [Corr_SGalpha750, Corr_SGalpha1p]
    tstat_W, p_val_W = stats.ranksums(Corr_SGalpha750, Corr_SGalpha1p)
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Alpha')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGalpha750))
    sig_n_two.append(len(Corr_SGalpha1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_SG_alpha['ms250Mean'], Corr_SG_alpha['ms500Mean'], \
                              Corr_SG_alpha['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_SG_alpha['ms1000Mean'], Corr_SG_alpha['s12Mean'], Corr_SG_alpha['s23Mean'], \
                          Corr_SG_alpha['s34Mean'], Corr_SG_alpha['s45Mean'],Corr_SG_alpha['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_SG_alpha = mean_diff
    Corr_P_SG_alpha = np.asarray(Corr_P_SG_alpha).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_SG_alpha, R_SG_alpha)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(mean_diff))
    comp_Wx_pVal_Corr.append(pVal)
    n_one.append(len(Corr_P_SG_alpha))
    n_two.append(len(R_SG_alpha))
    
    
   ##Beta
    Corr_SGbeta750 = np.array([Corr_SG_beta['ms250Mean'], Corr_SG_beta['ms500Mean'], \
                          Corr_SG_beta['ms750Mean']]).flatten()
    Corr_SGbeta750 = Corr_SGbeta750[~np.isnan(Corr_SGbeta750)]
    Corr_SGbeta1p = np.array([Corr_SG_beta['ms1000Mean'], Corr_SG_beta['s12Mean'], Corr_SG_beta['s23Mean'], \
                          Corr_SG_beta['s34Mean'], Corr_SG_beta['s45Mean'],Corr_SG_beta['s5pMean']]).flatten()
    Corr_SGbetaData = [Corr_SGbeta750, Corr_SGbeta1p]
    tstat_W, p_val_W = stats.ranksums(Corr_SGbeta750, Corr_SGbeta1p)
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGbeta750))
    sig_n_two.append(len(Corr_SGbeta1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_SG_beta['ms250Mean'], Corr_SG_beta['ms500Mean'], \
                              Corr_SG_beta['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_SG_beta['ms1000Mean'], Corr_SG_beta['s12Mean'], Corr_SG_beta['s23Mean'], \
                          Corr_SG_beta['s34Mean'], Corr_SG_beta['s45Mean'],Corr_SG_beta['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_SG_beta = mean_diff
    Corr_P_SG_beta = np.asarray(Corr_P_SG_beta).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_SG_beta, R_SG_beta)
    comp_Wx_pVal_Corr.append(pVal)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(Corr_P_SG_beta))
    n_one.append(len(Corr_P_SG_beta))
    n_two.append(len(R_SG_beta))
        
    #20Hz for Prestimulus comparison 
    Corr_SGbetapre = np.array(Corr_SG_beta['preStimMean'])
    Corr_SGbetapre = Corr_SGbetapre[~np.isnan(Corr_SGbetapre)]  #Select only 20Hz entries 
    Corr_SGbeta750_20 = np.array([Corr_SG_beta_20['ms250Mean'],Corr_SG_beta_20['ms500Mean'],\
                             Corr_SG_beta_20['ms750Mean']]).flatten()
    Corr_SGbeta750_20 = Corr_SGbeta750_20[~np.isnan(Corr_SGbeta750_20)]
    Corr_SGbeta1p_20 = np.array([Corr_SG_beta_20['ms1000Mean'], Corr_SG_beta_20['s12Mean'], \
                            Corr_SG_beta_20['s23Mean'], \
                Corr_SG_beta_20['s34Mean'], Corr_SG_beta_20['s45Mean'],Corr_SG_beta_20['s5pMean']]).flatten()
    Corr_SGbetaDataPre = [Corr_SGbetapre, Corr_SGbeta750_20, Corr_SGbeta1p_20]
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGbetapre, Corr_SGbeta750_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-750') 
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGbetapre))
    sig_n_two.append(len(Corr_SGbeta750_20))
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGbetapre, Corr_SGbeta1p_20)      
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-1p')  
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGbetapre))
    sig_n_two.append(len(Corr_SGbeta1p_20))
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGbeta750_20, Corr_SGbeta1p_20) 
        
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGbeta750_20))
    sig_n_two.append(len(Corr_SGbeta1p_20))
    
   ##Gamma 
    Corr_SGgamma750 = np.array([Corr_SG_gamma['ms250Mean'], Corr_SG_gamma['ms500Mean'], \
                           Corr_SG_gamma['ms750Mean']]).flatten()
    Corr_SGgamma750 = Corr_SGgamma750[~np.isnan(Corr_SGgamma750)]
    Corr_SGgamma1p = np.array([Corr_SG_gamma['ms1000Mean'],Corr_SG_gamma['s12Mean'], Corr_SG_gamma['s23Mean'], \
                    Corr_SG_gamma['s34Mean'], Corr_SG_gamma['s45Mean'],Corr_SG_gamma['s5pMean']]).flatten()
    Corr_SGgammaData = [Corr_SGgamma750, Corr_SGgamma1p]
    tstat_W, p_val_W = stats.ranksums(Corr_SGgamma750, Corr_SGgamma1p)
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGgamma750))
    sig_n_two.append(len(Corr_SGgamma1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_SG_gamma['ms250Mean'], Corr_SG_gamma['ms500Mean'], \
                              Corr_SG_gamma['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_SG_gamma['ms1000Mean'], Corr_SG_gamma['s12Mean'], Corr_SG_gamma['s23Mean'], \
                          Corr_SG_gamma['s34Mean'], Corr_SG_gamma['s45Mean'],Corr_SG_gamma['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_SG_gamma = mean_diff
    Corr_P_SG_gamma = np.asarray(Corr_P_SG_gamma).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_SG_gamma, R_SG_gamma)
    comp_Wx_pVal_Corr.append(pVal)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(P_SG_gamma))
    n_one.append(len(Corr_P_SG_gamma))
    n_two.append(len(R_SG_gamma))
    
    #40 & 60Hz for Prestimulus comparison 
    Corr_SGgammapre = np.array(Corr_SG_gamma['preStimMean'])
    Corr_SGgammapre = Corr_SGgammapre[~np.isnan(Corr_SGgammapre)] #Select only 40 & 60Hz entries 
    Corr_SGgamma750_4060 = np.array([Corr_SG_gamma_4060['ms250Mean'],Corr_SG_gamma_4060['ms500Mean'],\
                               Corr_SG_gamma_4060['ms750Mean']]).flatten()
    Corr_SGgamma750_4060 = Corr_SGgamma750_4060[~np.isnan(Corr_SGgamma750_4060)]
    Corr_SGgamma1p_4060 = np.array([Corr_SG_gamma_4060['ms1000Mean'], Corr_SG_gamma_4060['s12Mean'], \
                               Corr_SG_gamma_4060['s23Mean'], Corr_SG_gamma_4060['s34Mean'], \
                               Corr_SG_gamma_4060['s45Mean'],Corr_SG_gamma_4060['s5pMean']]).flatten()
    Corr_SGgammaDataPre = np.array([Corr_SGgammapre, Corr_SGgamma750_4060, Corr_SGgamma1p_4060])
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGgammapre, Corr_SGgamma750_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-750')     
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGgammapre))
    sig_n_two.append(len(Corr_SGgamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGgammapre, Corr_SGgamma750_4060)     
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGgammapre))
    sig_n_two.append(len(Corr_SGgamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(Corr_SGgamma750_4060, Corr_SGgamma1p_4060) 
    
    sig_Shank.append(name)
    sig_Layer.append('SG')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_SGgamma750_4060))
    sig_n_two.append(len(Corr_SGgamma1p_4060))
    
  ###GRANULAR 
   ##Alpha   
    Corr_Galpha750 = np.array([Corr_G_alpha['ms250Mean'], Corr_G_alpha['ms500Mean'], \
                          Corr_G_alpha['ms750Mean']]).flatten()
    Corr_Galpha750 = Corr_Galpha750[~np.isnan(Corr_Galpha750)]
    Corr_Galpha1p = np.array([Corr_G_alpha['ms1000Mean'], Corr_G_alpha['s12Mean'], Corr_G_alpha['s23Mean'], \
                          Corr_G_alpha['s34Mean'], Corr_G_alpha['s45Mean'],Corr_G_alpha['s5pMean']]).flatten()
    Corr_GalphaData = [Corr_Galpha750, Corr_Galpha1p]
    tstat_W, p_val_W = stats.ranksums(Corr_Galpha750, Corr_Galpha1p)
 
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Alpha')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Galpha750))
    sig_n_two.append(len(Corr_Galpha1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_G_alpha['ms250Mean'], Corr_G_alpha['ms500Mean'], \
                              Corr_G_alpha['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_G_alpha['ms1000Mean'],Corr_G_alpha['s12Mean'], Corr_G_alpha['s23Mean'], \
                          Corr_G_alpha['s34Mean'], Corr_G_alpha['s45Mean'],Corr_G_alpha['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_G_alpha = mean_diff
    Corr_P_G_alpha = np.asarray(Corr_P_G_alpha).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_G_alpha, R_G_alpha)
    comp_Wx_pVal_Corr.append(pVal)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(Corr_P_G_alpha))
    n_one.append(len(Corr_P_G_alpha))
    n_two.append(len(R_G_alpha))

   ##Beta   
    Corr_Gbeta750 = np.array([Corr_G_beta['ms250Mean'], Corr_G_beta['ms500Mean'], \
                         Corr_G_beta['ms750Mean']]).flatten()
    Corr_Gbeta750 = Corr_Gbeta750[~np.isnan(Corr_Gbeta750)]
    Corr_Gbeta1p = np.array([Corr_G_beta['ms1000Mean'], Corr_G_beta['s12Mean'], Corr_G_beta['s23Mean'], \
                          Corr_G_beta['s34Mean'], Corr_G_beta['s45Mean'],Corr_G_beta['s5pMean']]).flatten()
    Corr_GbetaData = [Corr_Gbeta750, Corr_Gbeta1p]
    tstat_W, p_val_W = stats.ranksums(Corr_Gbeta750, Corr_Gbeta1p)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Gbeta750))
    sig_n_two.append(len(Corr_Gbeta1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_G_beta['ms250Mean'], Corr_G_beta['ms500Mean'], \
                              Corr_G_beta['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_G_beta['ms1000Mean'], Corr_G_beta['s12Mean'], Corr_G_beta['s23Mean'], \
                          Corr_G_beta['s34Mean'], Corr_G_beta['s45Mean'],Corr_G_beta['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_G_beta = mean_diff
    Corr_P_G_beta = np.asarray(Corr_P_G_beta).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_G_beta, R_G_beta)
    comp_Wx_pVal_Corr.append(pVal)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(Corr_P_G_beta))
    n_one.append(len(Corr_P_G_beta))
    n_two.append(len(R_G_beta))
    
    #20Hz for Prestimulus comparison
    Corr_Gbetapre = np.array(Corr_G_beta['preStimMean'])
    Corr_Gbetapre = Corr_Gbetapre[~np.isnan(Corr_Gbetapre)]  #Select only 20Hz entries (all others are np.nan)
    Corr_Gbeta750_20 = np.array([Corr_G_beta_20['ms250Mean'],Corr_G_beta_20['ms500Mean'],\
                            Corr_G_beta_20['ms750Mean']]).flatten()
    Corr_Gbeta750_20 = Corr_Gbeta750_20[~np.isnan(Corr_Gbeta750_20)]
    Corr_Gbeta1p_20 =np.array([Corr_G_beta_20['ms1000Mean'],Corr_G_beta_20['s12Mean'],Corr_G_beta_20['s23Mean'], 
                     Corr_G_beta_20['s34Mean'], Corr_G_beta_20['s45Mean'],Corr_G_beta_20['s5pMean']]).flatten()
    Corr_GbetaDataPre = np.array([Corr_Gbetapre, Corr_Gbeta750_20, Corr_Gbeta1p_20])
    
    tstat_W, p_val_W = stats.ranksums(Corr_Gbetapre, Corr_Gbeta750_20) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-750') 
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Gbetapre))
    sig_n_two.append(len(Corr_Gbeta750_20))
    
    tstat_W, p_val_W = stats.ranksums(Corr_Gbetapre, Corr_Gbeta1p_20)     
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Gbetapre))
    sig_n_two.append(len(Corr_Gbeta1p_20))
    
    tstat_W, p_val_W = stats.ranksums(Corr_Gbeta750_20, Corr_Gbeta1p_20) 
        
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Beta-20')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Gbeta750_20))
    sig_n_two.append(len(Corr_Gbeta1p_20))

   ##Gamma 
    Corr_Ggamma750 = np.array([Corr_G_gamma['ms250Mean'], Corr_G_gamma['ms500Mean'], \
                          Corr_G_gamma['ms750Mean']]).flatten()
    Corr_Ggamma750 = Corr_Ggamma750[~np.isnan(Corr_Ggamma750)]
    Corr_Ggamma1p = np.array([Corr_G_gamma['ms1000Mean'], Corr_G_gamma['s12Mean'], Corr_G_gamma['s23Mean'], \
                          Corr_G_gamma['s34Mean'], Corr_G_gamma['s45Mean'],Corr_G_gamma['s5pMean']]).flatten()
    Corr_GgammaData = [Corr_Ggamma750, Corr_Ggamma1p]
    tstat_W, p_val_W = stats.ranksums(Corr_Ggamma750, Corr_Ggamma1p)
    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Ggamma750))
    sig_n_two.append(len(Corr_Ggamma1p))
    
    #Comparison between Pattern and Random
    early = np.nanmean(np.array([Corr_G_gamma['ms250Mean'], Corr_G_gamma['ms500Mean'], \
                              Corr_G_gamma['ms750Mean']]), axis=0) 
    late = np.nanmean(np.array([Corr_G_gamma['ms1000Mean'], Corr_G_gamma['s12Mean'], Corr_G_gamma['s23Mean'], \
                          Corr_G_gamma['s34Mean'], Corr_G_gamma['s45Mean'],Corr_G_gamma['s5pMean']]), axis=0)
    mean_diff = early - late 
    Corr_P_G_gamma = mean_diff
    Corr_P_G_gamma = np.asarray(Corr_P_G_gamma).flatten()
    stat, pVal = stats.wilcoxon(Corr_P_G_gamma, R_G_gamma)
    comp_Wx_pVal_Corr.append(pVal)
    comp_pattern_corr_mean1.append(np.mean(early))
    comp_pattern_corr_mean2.append(np.mean(late))
    comp_pattern_corr_diff.append(np.mean(Corr_P_G_gamma))
    n_one.append(len(Corr_P_G_gamma))
    n_two.append(len(R_G_gamma))
    
    
    #40 & 60Hz for Prestimulus comparison
    Corr_Ggammapre = np.array(Corr_G_gamma['preStimMean'])
    Corr_Ggammapre = Corr_Ggammapre[~np.isnan(Corr_Ggammapre)]  #Select only 40 & 60Hz entries 
    Corr_Ggamma750_4060 = np.array([Corr_G_gamma_4060['ms250Mean'],Corr_G_gamma_4060['ms500Mean'],\
                               Corr_G_gamma_4060['ms750Mean']]).flatten()
    Corr_Ggamma750_4060 = Corr_Ggamma750_4060[~np.isnan(Corr_Ggamma750_4060)]
    Corr_Ggamma1p_4060 = np.array([Corr_G_gamma_4060['ms1000Mean'],Corr_G_gamma_4060['s12Mean'], \
                              Corr_G_gamma_4060['s23Mean'], \
       Corr_G_gamma_4060['s34Mean'], Corr_G_gamma_4060['s45Mean'],Corr_G_gamma_4060['s5pMean']]).flatten()
    Corr_GgammaDataPre = np.array([Corr_Ggammapre, Corr_Ggamma750_4060, Corr_Ggamma1p_4060])

    tstat_W, p_val_W = stats.ranksums(Corr_Ggammapre, Corr_Ggamma750_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-750')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Ggammapre))
    sig_n_two.append(len(Corr_Ggamma750_4060))
    
    tstat_W, p_val_W = stats.ranksums(Corr_Ggammapre, Corr_Ggamma1p_4060)
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('pre-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Ggammapre))
    sig_n_two.append(len(Corr_Ggamma1p_4060))
    
    tstat_W, p_val_W = stats.ranksums(Corr_Ggamma750_4060, Corr_Ggamma1p_4060) 
                    
    sig_Shank.append(name)
    sig_Layer.append('G')
    sig_Band.append('Gamma-4060')
    sig_Protocol.append('P_Corr')
    sig_Comp.append('750-1p')
    sig_pVal_Wx.append(p_val_W)
    sig_n_one.append(len(Corr_Ggamma750_4060))
    sig_n_two.append(len(Corr_Ggamma1p_4060))
  
 ######################################################################################################   
####Plot comparisons

    #SG Alpha
    data = [Corr_SGalpha750, Corr_SGalpha1p, SGalpha750_R, SGalpha1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Supragranular_Alpha_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 0.5e-9, outputpath)
    
    #SG Beta
    data = [Corr_SGbeta750, Corr_SGbeta1p, SGbeta750_R, SGbeta1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Supragranular_Beta_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 0.8e-10, outputpath)
    
    #SG Gamma 
    data = [Corr_SGgamma750, Corr_SGgamma1p, SGgamma750_R, SGgamma1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Supragranular_Gamma_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 1.0e-11, outputpath)
    
    #G Alpha
    data = [Corr_Galpha750, Corr_Galpha1p, Galpha750_R, Galpha1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Granular_Alpha_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 0.7e-9, outputpath)
    
    #G Beta
    data = [Corr_Gbeta750, Corr_Gbeta1p, Gbeta750_R, Gbeta1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Granular_Beta_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 1.0e-10, outputpath)
    
    #G Gamma 
    data = [Corr_Ggamma750, Corr_Ggamma1p, Ggamma750_R, Ggamma1p_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Granular_Gamma_Comp-PCvsR_' + str(name) + '.png'
    plot_boxplot_tight_double(data, 1.0e-11, outputpath)

    
    #SG 20Hz
    data = [Corr_SGbetaDataPre, SGbetaDataPre_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Supragranular_Beta_Comp-PCvsR_Pre_' + str(name) + '.png'
    plot_boxplot_tight_triple(data, 0.8e-10, outputpath)
    
    #SG 40,60Hz
    data = [Corr_SGgammaDataPre, SGgammaDataPre_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Supragranular_Gamma_Comp-PCvsR_Pre_' + str(name) + '.png'
    plot_boxplot_tight_triple(data, 1.2e-11, outputpath)
    
    #G 20Hz
    data = [Corr_GbetaDataPre, GbetaDataPre_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Granular_Beta_Comp-PCvsR_Pre_' + str(name) + '.png'
    plot_boxplot_tight_triple(data, 1.2e-10, outputpath)
    
    #G 40,60Hz
    data = [Corr_GgammaDataPre, GgammaDataPre_R]
    outputpath = path +'/Analysis/DistancefromProbe/PSD_Granular_Gamma_Comp-PCvsR_Pre_' + str(name) + '.png'
    plot_boxplot_tight_triple(data, 1.3e-11, outputpath)


 ######################################################################################################     
####Save significance data to csv
    sig_Shank = np.asarray(sig_Shank)
    sig_Layer = np.asarray(sig_Layer)
    sig_Band = np.asarray(sig_Band)
    sig_Protocol = np.asarray(sig_Protocol)
    sig_Comp = np.asarray(sig_Comp)
    sig_pVal_Wx = np.asarray(sig_pVal_Wx)
    sig_n_one = np.asarray(sig_n_one)
    sig_n_two = np.asarray(sig_n_two)
    
    datatofile = np.array([sig_Shank, sig_Layer, sig_Band, sig_Protocol, sig_Comp, sig_pVal_Wx, \
                          sig_n_one, sig_n_two]).T
    headers = np.array(['Shank', 'Layer', 'Band','Protocol','Comparison','pVal_WilcoxonRankSum', 'n1', 'n2'])
    
    pd.DataFrame(datatofile).to_csv(path +'/Analysis/DistancefromProbe/PowerSignificances_DistanceProbe'\
                                    +name+'.csv', header = headers)    

######################################################################################################
####Save Pattern vs Random significance data to csv
    comp_layer = np.asarray(comp_layer)
    comp_band = np.asarray(comp_band)
    comparison = np.asarray(comparison)
    comp_pattern_mean1 = np.asarray(comp_pattern_mean1)
    comp_pattern_mean2 = np.asarray(comp_pattern_mean2)
    comp_pattern_diff = np.asarray(comp_pattern_diff) 
    comp_random_mean1 = np.asarray(comp_random_mean1)
    comp_random_mean2 = np.asarray(comp_random_mean2)
    comp_random_diff = np.asarray(comp_random_diff)
    comp_pattern_corr_mean1 = np.asarray(comp_pattern_corr_mean1)
    comp_pattern_corr_mean2 = np.asarray(comp_pattern_corr_mean2)
    comp_pattern_corr_diff = np.asarray(comp_pattern_corr_diff)
    comp_Wx_pVal_Corr = np.asarray(comp_Wx_pVal_Corr)
    comp_Wx_pVal = np.asarray(comp_Wx_pVal)
    n_one = np.asarray(n_one)
    n_two = np.asarray(n_two)
    
    datatofile = np.array([comp_layer, comp_band, comparison, comp_pattern_mean1,comp_pattern_mean2,\
                    comp_pattern_diff, comp_random_mean1, comp_random_mean2, comp_random_diff, comp_Wx_pVal,\
              comp_pattern_corr_mean1 ,comp_pattern_corr_mean2, comp_pattern_corr_diff, comp_Wx_pVal_Corr, \
                           n_one, n_two]).T
    
    headers = np.array(['Layer', 'Band', 'Comparison','PatternMean1','PatternMean2', 'PatternDiff',\
                       'RandomMean1','RandomMean2', 'RandomDiff', 'pVal_Wilcoxon_PvsR',\
                        'PatternCorrMean1','PatternCorrMean2', 'PatternCorrDiff', 'pVal_Wilcoxon_PCorrvsR', \
                       'n1PcorrvsR', 'n2PcorrvsR'])
    
    pd.DataFrame(datatofile).to_csv(path +'/Analysis/DistancefromProbe/PowerSignificances_DistanceProbe_PatternRandomComp'\
                                    +name+'.csv', header = headers)   
    
######################################################################################################  