"""
Gwendolyn English 04.08.2021

Functions for the MUA analysis between all animals within an experimental set 
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

def pullData_MUA(filepath, trials, shank):
    """
    This function reads the comiled data file and extracts relevant data required for specified analysis.
    Inputs: filepath, trials ("probe", "context"), Shank (0, 1)
    Outputs: Pandas dataframe with corresponding data
    """
    #Load dataframe
    data = pickle.load(open(filepath, 'rb'))
    time = data['time']
    del data['time']
    df = pd.DataFrame(data)
    
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
    
    #Separate data into 'pattern' and 'random' 
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']
            
    #Reindex
    df1 = df1.reset_index(drop = True)
    df2 = df2.reset_index(drop = True)

    #Pull only channels considered units with glass Delta index > 0.5     
    df1_nonUnits = df1.index[df1['unit'] ==False].tolist()
    df2_nonUnits = df2.index[df2['unit'] ==False].tolist()
    nonUnits = list(set(df1_nonUnits) & set(df2_nonUnits))
    #Remove non-units
    df1 = df1.drop(index = nonUnits)
    df2 = df2.drop(index = nonUnits)
  
    #Reindex
    df1 = df1.reset_index(drop = True)
    df2 = df2.reset_index(drop = True)
    
    return df1, df2     
        
#for peak firing rate comparisons  
def trialbytrial_MUA(df1, df2, name, stim, path):    
    #Trial-by-trial ttest holders
    fr_ttest = []
    fr_P = []
    fr_R = [] 
    
    #Full Firing Rate Pattern and Random
    fullFR_P = []
    fullFR_R = [] 
    
    layer = []
    animalID = []
    
    #Create MUA Analysis Folder
    if not os.path.exists(path + '/Analysis/MUA'):
        os.mkdir(path + '/Analysis/MUA')
    
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        layer.append(df1.loc[row, 'layer'])
        animalID.append(df1.loc[row, 'animalID'])

        #convert to firing rates and extract peaks 
        data1 = np.amax(firing_rate_ms_bins_trial_nonsmoothed(df1.loc[row, 'ms_bins']), axis = 1)
        data2 = np.amax(firing_rate_ms_bins_trial_nonsmoothed(df2.loc[row, 'ms_bins']), axis = 1)
        
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
        datafullFR1 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(df1.loc[row, 'ms_bins']), axis = 0)
        datafullFR2 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(df2.loc[row, 'ms_bins']), axis = 0)
        
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
    scatter2D_edgelabels_staticAxes(fr_P,fr_R,fr_ttest, path + '/Analysis/MUA/' + \
                 'Scatter_FiringRate_mean_'+ name + '_' + stim + '.png','FR', corrected_sig)
    
    #Write data to csv    
    datatofile = np.array([animalID, layer, fr_P, fr_R, fr_ttest, corrected_sig, bh_val])
    datatofile = datatofile.T
    
    headers = np.array(['Animal', 'Layer', 'FR_P_mean', 'FR_R_mean','p_Val', 'Sig_afterCorr', 'BH_value'])
    
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/MUA/' + 'FiringRate_ttestResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
    
    #Plot Cortical Layer Ratios      
    #Create new array with necessary data  
    ratio = (fr_P - fr_R) / (fr_P + fr_R) 
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio': ratio, 'compLayer': compLayer, 'fr_R': fr_R, 'fr_P': fr_P})
    #Remove outliers
    df = df.drop(df[df.ratio > 0.5].index)
    df = df.drop(df[df.ratio < -0.5].index)
    #Create plot
    plot_compact_laminar_profile(df, path + '/Analysis/MUA/' + 'Layer_FiringRate_' + name + '_' + stim + '.png', \
                                 stim)
    
    #Layer-wise Wilcoxon signed-rank tests 
    pVals_rel_Wx = [] 
    conf_low = []
    conf_high = []
    median_P = []
    median_R = []
    ns = []
    
    #SG - L2/3
    P_SG = df[df['compLayer'] == 'SG']['fr_P']
    R_SG = df[df['compLayer'] == 'SG']['fr_R']
    stat, pVal = stats.wilcoxon(P_SG, R_SG)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_SG, R_SG, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_SG))
    median_R.append(np.median(R_SG))
    ns.append(len(P_SG))
        
    #G - L4
    P_G = df[df['compLayer'] == 'G']['fr_P']
    R_G = df[df['compLayer'] == 'G']['fr_R']
    stat, pVal = stats.wilcoxon(P_G, R_G)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_G, R_G, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_G))
    median_R.append(np.median(R_G))
    ns.append(len(P_G))
   
    #IGU - L5 
    P_IGU = df[df['compLayer'] == 'IGU']['fr_P']
    R_IGU = df[df['compLayer'] == 'IGU']['fr_R']
    stat, pVal = stats.wilcoxon(P_IGU, R_IGU)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_IGU, R_IGU, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGU))
    median_R.append(np.median(R_IGU))
    ns.append(len(P_IGU))

    #IGL - L6 
    P_IGL = df[df['compLayer'] == 'IGL']['fr_P']
    R_IGL = df[df['compLayer'] == 'IGL']['fr_R']
    stat, pVal = stats.wilcoxon(P_IGL, R_IGL)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_IGL, R_IGL, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGL))
    median_R.append(np.median(R_IGL))
    ns.append(len(P_IGL))

    
    #Write statistics to csv file
    datatofile = np.array([['SG','G','IGU','IGL'], pVals_rel_Wx, conf_low, conf_high, \
                          median_P, median_R, ns])
    datatofile = datatofile.T
    headers = (['Layer','pVal_Wilcoxon','conf_Low','conf_High', 'med_P,', 'med_R', 'n'])
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/MUA/' + 'LayerwiseFiringRate_WilcoxonResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
    
####Plot Average Firing Rates by layer
    compLayer = laminar_labelTolayer(layer)
    
    #Supragranular Layer - L2/3
    sg = np.asarray(np.where(compLayer == 'SG')).flatten()
    SG_P = np.take(fullFR_P, sg, axis = 0)
    SG_R = np.take(fullFR_R, sg, axis = 0)
    plot_fr_grandAvg(SG_P, SG_R, path + '/Analysis/MUA/', name, stim, 'SG')
    
    #Granular Layer - L4
    g = np.asarray(np.where(compLayer == 'G')).flatten()
    G_P = np.take(fullFR_P, g, axis = 0)
    G_R = np.take(fullFR_R, g, axis = 0)
    plot_fr_grandAvg(G_P, G_R, path + '/Analysis/MUA/', name, stim, 'G')
    
    #Infragranular Upper Layer - L5
    igu = np.asarray(np.where(compLayer == 'IGU')).flatten()
    IGU_P = np.take(fullFR_P, igu, axis = 0)
    IGU_R = np.take(fullFR_R, igu, axis = 0)
    plot_fr_grandAvg(IGU_P, IGU_R, path + '/Analysis/MUA/', name, stim, 'IGU')
    
    #Infragranular Lower Layer - L6
    igl = np.asarray(np.where(compLayer == 'IGL')).flatten()
    IGL_P = np.take(fullFR_P, igl, axis = 0)
    IGL_R = np.take(fullFR_R, igl, axis = 0)
    plot_fr_grandAvg(IGL_P, IGL_R, path + '/Analysis/MUA/', name, stim, 'IGL')
    
    
def trialbytrial_MUA_thalamus(df1, df2, name, stim, path):    
    #Trial-by-trial ttest holders
    fr_ttest = []
    fr_P = []
    fr_R = [] 
    
    #Full Firing Rate Pattern and Random
    fullFR_P = []
    fullFR_R = [] 
    
    layer = []
    animalID = []
    
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        layer.append(df1.loc[row, 'layer'])
        animalID.append(df1.loc[row, 'animalID'])

        #convert to firing rates and extract peaks 
        data1 = np.amax(firing_rate_ms_bins_trial_nonsmoothed(df1.loc[row, 'ms_bins']), axis = 1)
        data2 = np.amax(firing_rate_ms_bins_trial_nonsmoothed(df2.loc[row, 'ms_bins']), axis = 1)
        
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
        datafullFR1 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(df1.loc[row, 'ms_bins']), axis = 0)
        datafullFR2 = np.nanmean(firing_rate_ms_bins_trial_nonsmoothed(df2.loc[row, 'ms_bins']), axis = 0)
        
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
    scatter2D_edgelabels_staticAxes(fr_P,fr_R,fr_ttest, path + '/Analysis/MUA/' + \
                 'Scatter_FiringRate_mean_'+ name + '_' + stim + '.png','FR', corrected_sig)
    
    #Write data to csv    
    datatofile = np.array([animalID, layer, fr_P, fr_R, fr_ttest, corrected_sig, bh_val])
    datatofile = datatofile.T
    
    headers = np.array(['Animal', 'Layer', 'FR_P_mean', 'FR_R_mean','p_Val', 'Sig_afterCorr', 'BH_value'])
    
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/MUA/' + 'FiringRate_ttestResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
 
    #Plot Thalamic Region Ratios
    #Compute ratios 
    ratio = (fr_P - fr_R) / (fr_P + fr_R) 
    shank_label = np.repeat(name, len(ratio))
    df = pd.DataFrame({'ratio': ratio, 'label': shank_label, 'fr_R': fr_R, 'fr_P': fr_P})
    #Remove outliers
    df = df.drop(df[df.ratio > 0.5].index)
    df = df.drop(df[df.ratio < -0.5].index)
    #Create Plot
    plot_thalamus_ratios(df, path + '/Analysis/MUA/' + 'Thalamus_FiringRate_'+name+'_'+stim+'.png', stim)
    #Wilcoxon signed-rank tests 
    stat, pVal = stats.ttest_rel(fr_P, fr_R)
    datatofile = [pVal]
    headers = (['pVal_Wilcoxon'])
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/MUA/' + 'Region_WilcoxonResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
     