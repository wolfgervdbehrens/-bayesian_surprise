"""
Gwendolyn English 04.08.2021

Functions for the LFP analysis between all animals within an experimental set
"""

##########################################################################################################
#Import required packages & functions
import pickle
import sys, os, numpy as np, pandas as pd
from scipy import stats
from scipy import signal

from plotting import * 
from helper_functions import *
##########################################################################################################

def pullData_LFP(filepath, trials, shank):
    """
    This function reads the compiled data file and extracts relevant data required for specified analysis.
    Inputs: filepath, Trials ('probe', 'context'), shank (0,1)
    Outputs: Pandas dataframe with corresponding data
    """
    #Load data
    data = pickle.load(open(filepath, 'rb'))
    
    trialData = {'animalID':data['animalID'],  'shank':data['shank'],  'layer':data['layer'], \
                 'trode':data['trode'], 'paradigm':data['paradigm'],  'ketID':data['ketID'], \
                 'whiskerID':data['whiskerID'],'ERP':data['ERP'], 'PSD':data['PSD'], 'prestimPSD':data['prestimPSD']}
    df = pd.DataFrame(data=trialData)
    
    #Establish trials to pull
    if trials == 'probe':
        extData = df[df['whiskerID'] == 'C1']
    if trials == 'context':
        extData = df[df['whiskerID'].isin(['B1', 'C2', 'D1'])]
    df = extData
    
    #Establish shank to pull
    if shank == 0:
        extData = df[df['shank'] == 0]
    if shank == 1:
        extData = df[df['shank'] == 1]
    df = extData
    
    #Separate Pattern and Random
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']
        
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    return (df1, df2)


def trialbytrial_LFP(path, df1, df2, name, stim, state):
    
    #Trial-by-trial holders
    ERP_ttest = []
    ERP_P = []
    ERP_R = []
    ERP_late_ttest = []
    ERP_late_P = []
    ERP_late_R = []
    ERP_P_traces = []
    ERP_R_traces = []
    layer = []
    animal = [] 
    
    #Create Analysis Folder
    if not os.path.exists(path + '/Analysis/LFP'):
        os.mkdir(path + '/Analysis/LFP')
    
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons
        arr1 = [df1.loc[row,'animalID'],df1.loc[row,'shank'],df1.loc[row,'layer'], df1.loc[row,'whiskerID']]
        arr2 = [df2.loc[row,'animalID'], df2.loc[row,'shank'], df2.loc[row,'layer'], df2.loc[row,'whiskerID']]
        if arr1 != arr2:
            print('Error!')
        layer.append(df1.loc[row, 'layer'])
        animal.append(df1.loc[row, 'animalID'])
        
        #Early Window
        data1 = np.amin(np.asarray(df1.loc[row, 'ERP'])[:, 50:150], axis=1)
        data2 = np.amin(np.asarray(df2.loc[row, 'ERP'])[:, 50:150], axis=1)
        
        #Remove any nan entries
        data1nans = np.argwhere(np.isnan(data1)).flatten()
        data2nans = np.argwhere(np.isnan(data2)).flatten()
        naninds = np.hstack((data1nans, data2nans)).flatten()
        naninds = naninds.astype(int)
        data1 = np.delete(data1, naninds)
        data2 = np.delete(data2, naninds)
        
        tstat, pval = stats.ttest_rel(data1, data2)
        ERP_ttest.append(pval)
        ERP_P.append(np.mean(data1))
        ERP_R.append(np.mean(data2))
        
        #Late Window
        if state == 'anae':
            data1 = np.amin(np.asarray(df1.loc[row, 'ERP'])[:, 150:250], axis=1)
            data2 = np.amin(np.asarray(df2.loc[row, 'ERP'])[:, 150:250], axis=1) 
        if state == 'awake':
            data1 = np.amin(np.asarray(df1.loc[row, 'ERP'])[:, 150:300], axis=1) 
            data2 = np.amin(np.asarray(df2.loc[row, 'ERP'])[:, 150:300], axis=1) 
        tstat, pval = stats.ttest_rel(data1, data2)
        ERP_late_ttest.append(pval)
        ERP_late_P.append(np.nanmean(data1))
        ERP_late_R.append(np.nanmean(data2))

        #Full ERP traces
        traceP = np.nanmean(df1.loc[row, 'ERP'], axis=0)
        traceR = np.nanmean(df2.loc[row, 'ERP'], axis=0)
        ERP_P_traces.append(traceP)
        ERP_R_traces.append(traceR)
    
    #Lists to arrays
    animals = np.asarray(animal) 
    compLayer = laminar_labelTolayer(df1['layer'])
    ERP_P_traces = np.asarray(ERP_P_traces)
    ERP_R_traces = np.asarray(ERP_R_traces)
    
    #Supragranular layer
    ind = np.where(compLayer == 'SG')
    SG_P = np.take(ERP_P_traces, ind, axis=0)
    SG_R = np.take(ERP_R_traces, ind, axis=0)
    SG_P = np.reshape(SG_P, (np.shape(SG_P)[1], np.shape(SG_P)[2]))
    SG_R = np.reshape(SG_R, (np.shape(SG_R)[1], np.shape(SG_R)[2]))
    plot_evoked_channels_sem(SG_P, SG_R, path + '/Analysis/LFP/LFP-Trace_SG_'+name+'_'+stim + '.png')
    if stim == 'probe': limits = [-100, 50]
    if stim == 'context': limits = [-15,15]    
    limits = 0
    plot_diff_sem(SG_P, SG_R, path + '/Analysis/LFP/LFP-Trace_SG_diff_'+name+ '_' + stim + '.png', stim, limits)
    
    #Granular layer
    ind = np.where(compLayer == 'G')
    G_P = np.take(ERP_P_traces, ind, axis=0)
    G_R = np.take(ERP_R_traces, ind, axis=0)
    G_P = np.reshape(G_P, (np.shape(G_P)[1], np.shape(G_P)[2]))
    G_R = np.reshape(G_R, (np.shape(G_R)[1], np.shape(G_R)[2]))
    plot_evoked_channels_sem(G_P,G_R, path + '/Analysis/LFP/LFP-Trace_G_'+name+'_' + stim + '.png')
    if stim == 'probe': limits = [-100, 50]
    if stim == 'context': limits = [-15,15]
    limits = 0
    plot_diff_sem(G_P,G_R, path + '/Analysis/LFP/LFP-Trace_G_diff_' + name + '_' + stim + '.png', stim, limits)
    
    #Upper Infragranular Layer 
    ind = np.where(compLayer == 'IGU')
    IGU_P = np.take(ERP_P_traces, ind, axis=0)
    IGU_R = np.take(ERP_R_traces, ind, axis=0)
    IGU_P = np.reshape(IGU_P, (np.shape(IGU_P)[1], np.shape(IGU_P)[2]))
    IGU_R = np.reshape(IGU_R, (np.shape(IGU_R)[1], np.shape(IGU_R)[2]))
    plot_evoked_channels_sem(IGU_P,IGU_R, path + '/Analysis/LFP/LFP-Trace_IGU_'+name+'_'+stim+ '.png')
    if stim == 'probe': limits = [-50, 50]
    if stim == 'context': limits = [-15,15]   
    limits = 0
    plot_diff_sem(IGU_P,IGU_R,  path + '/Analysis/LFP/LFP-Trace_IGU_diff_'+name+'_'+stim + '.png', stim, limits)
    
    #Lower Infragranular Layer 
    ind = np.where(compLayer == 'IGL')
    IGL_P = np.take(ERP_P_traces, ind, axis=0)
    IGL_R = np.take(ERP_R_traces, ind, axis=0)
    IGL_P = np.reshape(IGL_P, (np.shape(IGL_P)[1], np.shape(IGL_P)[2]))
    IGL_R = np.reshape(IGL_R, (np.shape(IGL_R)[1], np.shape(IGL_R)[2]))
    plot_evoked_channels_sem(IGL_P,IGL_R, path + '/Analysis/LFP/LFP-Trace_IGL_'+name+'_'+stim+ '.png')
    if stim == 'probe': limits = [-50, 50]
    if stim == 'context': limits = [-15,15]   
    limits = 0
    plot_diff_sem(IGL_P, IGL_R, path + '/Analysis/LFP/LFP-Trace_IGL_diff_'+name+'_'+stim+'.png', stim, limits)
    
    ERP_P = np.asarray(ERP_P) * -1000000
    ERP_R = np.asarray(ERP_R) * -1000000
    ERP_ttest = np.asarray(ERP_ttest)
    ERP_late_P = np.asarray(ERP_late_P) * -1000000
    ERP_late_R = np.asarray(ERP_late_R) * -1000000
    ERP_late_ttest = np.asarray(ERP_late_ttest)
    layer = np.asarray(layer)

    ratio = (ERP_P - ERP_R) / (ERP_R + ERP_P)
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio':ratio,  'compLayer':compLayer, 'ERP_P': ERP_P, 'ERP_R': ERP_R})
    plot_compact_laminar_profile(df, path + '/Analysis/LFP/Layer_LFP_'+name+'_'+stim+'.png', stim)
    #Mean trace data to file 
    pickle.dump({'ratio':ratio, 'compLayer':compLayer, 'ERP_P': ERP_P, 'ERP_R': ERP_R}, 
    open(path + '/Analysis/LFP/LFP-NSI_' + name + '_' + stim + '.pickle', 'wb'), protocol =-1)
   
    #Layer-wise Wilcoxon tests 
    pVals_rel_Wx = [] 
    median_P = []
    median_R = []
    conf_low = []
    conf_high = []
    ns = []
    
    #SG
    P_SG = df[df['compLayer'] == 'SG']['ERP_P']
    R_SG = df[df['compLayer'] == 'SG']['ERP_R']
    stat, pVal = stats.wilcoxon(P_SG, R_SG)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_SG, R_SG, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_SG))
    median_R.append(np.median(R_SG))
    ns.append(len(P_SG))

    #G
    P_G = df[df['compLayer'] == 'G']['ERP_P']
    R_G = df[df['compLayer'] == 'G']['ERP_R']
    stat, pVal = stats.wilcoxon(P_G, R_G)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_G, R_G, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_G))
    median_R.append(np.median(R_G))
    ns.append(len(P_G))
    
    #IGU
    P_IGU = df[df['compLayer'] == 'IGU']['ERP_P']
    R_IGU = df[df['compLayer'] == 'IGU']['ERP_R']
    stat, pVal = stats.wilcoxon(P_IGU, R_IGU)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_IGU, R_IGU, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGU))
    median_R.append(np.median(R_IGU))
    ns.append(len(P_IGU))

    #IGL
    P_IGL = df[df['compLayer'] == 'IGL']['ERP_P']
    R_IGL = df[df['compLayer'] == 'IGL']['ERP_R']
    stat, pVal = stats.wilcoxon(P_IGL, R_IGL)
    pVals_rel_Wx.append(pVal)
    #ci = non_param_paired_CI(P_IGL, R_IGL, .95)
    ci = ['na','na'] #Awake
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGL))
    median_R.append(np.median(R_IGL))
    ns.append(len(P_IGL))
    
    #Write statistics to csv file
    datatofile = np.array([['SG','G','IGU','IGL'], pVals_rel_Wx, median_P, median_R, \
                           conf_low, conf_high, ns])
    datatofile = datatofile.T
    headers = (['Layer', 'pVals_Wilcoxon','med_P','med_R','conf_Low','conf_High', 'n'])
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/LFP/LayerwiseLFP_WilcoxonTestResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
####Late Window     
    ratio = (ERP_late_P - ERP_late_R) / (ERP_late_R + ERP_late_P)
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio':ratio,  'compLayer':compLayer, 'ERP_P': ERP_late_P, \
                      'ERP_R': ERP_late_R})
    plot_compact_laminar_profile(df, path + '/Analysis/LFP/Layer_LFP_LateWindow_' + name + '_' + stim + '.png', stim)
    #Mean trace data to file 
    pickle.dump({'ratio':ratio,  'compLayer':compLayer, 'ERP_P': ERP_P, 'ERP_R': ERP_R}, 
    open(path + '/Analysis/LFP/LFP-NSI-LateWindow_' + name + '_' + stim + '.pickle', 'wb'), protocol =-1)
    
    #Layer-wise relative ttests
    pVals_rel_Wx = [] 
    median_P = []
    median_R = []
    conf_low = []
    conf_high = []
    
    #SG
    P_SG = df[df['compLayer'] == 'SG']['ERP_P']
    R_SG = df[df['compLayer'] == 'SG']['ERP_R']
    stat, pVal = stats.wilcoxon(P_SG, R_SG)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_SG, R_SG, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_SG))
    median_R.append(np.median(R_SG))

    #G
    P_G = df[df['compLayer'] == 'G']['ERP_P']
    R_G = df[df['compLayer'] == 'G']['ERP_R']
    stat, pVal = stats.wilcoxon(P_G, R_G)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(P_G, R_G, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_G))
    median_R.append(np.median(R_G))

    #IGU
    P_IGU = df[df['compLayer'] == 'IGU']['ERP_P']
    R_IGU = df[df['compLayer'] == 'IGU']['ERP_R']
    stat, pVal = stats.wilcoxon(P_IGU, R_IGU)
    pVals_rel_Wx.append(pVal) 
    ci = non_param_paired_CI(P_IGU, R_IGU, .95)
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGU))
    median_R.append(np.median(R_IGU))
        
    #IGL
    P_IGL = df[df['compLayer'] == 'IGL']['ERP_P']
    R_IGL = df[df['compLayer'] == 'IGL']['ERP_R']
    stat, pVal = stats.wilcoxon(P_IGL, R_IGL)
    pVals_rel_Wx.append(pVal)
    if state == "anae":
        ci = non_param_paired_CI(P_IGL, R_IGL, .95)
    if state == "awake":
        ci = ['na','na']
    conf_low.append(ci[0])
    conf_high.append(ci[1])
    median_P.append(np.median(P_IGL))
    median_R.append(np.median(R_IGL))
 
    
    #Write statistics to csv file
    datatofile = np.array([['SG','G','IGU','IGL'], pVals_rel_Wx, median_P, median_R, \
                           conf_low, conf_high])
    datatofile = datatofile.T
    headers = (['Layer', 'pVals_Wilcoxon', 'med_P', 'med_R', 'conf_Low', 'conf_High'])
    pd.DataFrame(datatofile).to_csv(path + '/Analysis/LFP/LayerwiseLFP-late_ttestResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
    
    #Mean trace data to file 
    pickle.dump({'G_P': G_P, 'SG_P': SG_P, 'IGU_P': IGU_P, 'IGL_P': IGL_P, 
                 'G_R': G_R, 'SG_R': SG_R, 'IGU_R': IGU_R, 'IGL_R': IGL_R}, 
   open(path + '/Data/FinalLFPTraces_' + name + '_' + stim + '.pickle', 'wb'), protocol =-1)

