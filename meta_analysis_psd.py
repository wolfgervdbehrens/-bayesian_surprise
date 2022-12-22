"""
Gwendolyn English 15.11.2021

Functions for the PSD nalysis between all animals within an experimental set 
"""

##########################################################################################################
#Import required packages & functions
import sys
import os
import numpy as np
import pandas as pd  
from scipy import stats
from scipy import signal 

from plotting import * 
from helper_functions import *
##########################################################################################################

def pullDataPSD(filepath, trials, shank, window):
    """
    This function reads the compiled data file and extracts relevant data required for specified analysis.
    Inputs: Filepath, Trials ('probe', 'context'), Shank (0,1), Window ('pre', 'post'). 
    Outputs: Pandas dataframe with corresponding data
    """
    #Load data
    data = pickle.load(open(filepath, 'rb'))
    
    #Select PSD frequencies 
    if window == 'pre': freqs = data['prestimPSD_freqs']
    if window == 'post': freqs = data['PSD_freqs']

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
    
    #Separate Pattern and Random data 
    df1 = df[df['paradigm'] == 'P']
    df2 = df[df['paradigm'] == 'R']
        
    df1 = df1.reset_index(drop = True)
    df2 = df2.reset_index(drop = True)

    return df1, df2, freqs


def trialbytrialPSD(path, df1, df2, name, stim, window, freqs):    
    #Trial-by-trial ttest holders
    #Frequency bins collapsed into specific bands 
    col_alpha_ttest = []
    col_alpha_P = []
    col_alpha_R = [] 
    
    col_beta_ttest = []
    col_beta_P = []
    col_beta_R = [] 
    
    col_gamma_ttest = [] 
    col_gamma_P = []
    col_gamma_R = []
  
    #Full PSD from 8-60Hz (alpha: 8-12, beta: 16-28, gamma: 32-60)
    psd_P = []
    psd_R = []
    
    #Meta data 
    layer = []
    animal = []
    
#####Cycle data 
    for row in range(0, len(df1.index)):
        
        #Ensure proper comparisons 
        arr1 = [df1.loc[row, 'animalID'],df1.loc[row, 'shank'],df1.loc[row, 'layer'],df1.loc[row, 'whiskerID']]
        arr2 = [df2.loc[row, 'animalID'],df2.loc[row, 'shank'],df2.loc[row, 'layer'],df2.loc[row, 'whiskerID']]
        if arr1 != arr2:
             print('Error!')
        layer.append(df1.loc[row, 'layer'])
        animal.append(df1.loc[row, 'animalID'])
        
        if window == 'pre':
            data1 = np.asarray(df1.loc[row, 'preStim_PSD'])
            data2 = np.asarray(df2.loc[row, 'preStim_PSD'])
        if window == 'post':
            data1 = np.asarray(df1.loc[row, 'PSD'])
            data2 = np.asarray(df2.loc[row, 'PSD'])

        #Remove any nan entries
        data1nans = np.argwhere(np.isnan(data1[:,0])).flatten()
        data2nans = np.argwhere(np.isnan(data2[:,0])).flatten()
        naninds = np.hstack((data1nans, data2nans)).flatten()
        naninds = naninds.astype(int)
        data1 = np.delete(data1, naninds, axis = 0)
        data2 = np.delete(data2, naninds, axis = 0)
    
        #Append full PSD from alpha to gamma
        indices = np.where(np.in1d(freqs, [8,12,16,20,24,28,32,36,40,44,48,52,56,60]))[0]
        psd_P.append(np.mean(np.take(data1, indices, axis = 1), axis =0))
        psd_R.append(np.mean(np.take(data2, indices, axis = 1), axis =0))
    
        #Select alpha data 
        indices = np.where(np.in1d(freqs, [8,12]))[0]
        alpha1 = np.take(data1, indices, axis = 1)             
        alpha2 = np.take(data2, indices, axis = 1)
        
        ind_alpha_P = []
        ind_alpha_R = []
        for entry in np.arange(2): 
            alpha1freq = alpha1[:,entry]
            alpha2freq = alpha2[:,entry]
            ind_alpha_P.append(alpha1freq)
            ind_alpha_R.append(alpha2freq) 
        ind_alpha_P = np.sum(ind_alpha_P, axis = 1)    
        ind_alpha_R = np.sum(ind_alpha_R, axis = 1)
        tstat, pval = stats.ttest_rel(ind_alpha_P, ind_alpha_R) 
        col_alpha_ttest.append(pval)
        col_alpha_P.append(np.median(ind_alpha_P))
        col_alpha_R.append(np.median(ind_alpha_R)) 
        
        #Select beta data 
        indices = np.where(np.in1d(freqs, [16,20,24,28]))[0]
        beta1 = np.take(data1, indices, axis = 1)             
        beta2 = np.take(data2, indices, axis = 1)
       
        ind_beta_P = []
        ind_beta_R = []
        for entry in np.arange(4): 
            beta1freq = beta1[:,entry]
            beta2freq = beta2[:,entry]
            ind_beta_P.append(beta1freq)
            ind_beta_R.append(beta2freq)
        ind_beta_P = np.sum(ind_beta_P, axis = 1)    
        ind_beta_R = np.sum(ind_beta_R, axis = 1)
        tstat, pval = stats.ttest_rel(ind_beta_P, ind_beta_R) 
        col_beta_ttest.append(pval)
        col_beta_P.append(np.median(ind_beta_P))
        col_beta_R.append(np.median(ind_beta_R)) 
    
        #Select gamma data 
        indices = np.where(np.in1d(freqs, [32,36,40,44,48,52,56,60]))[0]
        gamma1 = np.take(data1, indices, axis = 1)             
        gamma2 = np.take(data2, indices, axis = 1)
        
        ind_gamma_P = []
        ind_gamma_R = []
        for entry in np.arange(8): 
            gamma1freq = gamma1[:,entry]
            gamma2freq = gamma2[:,entry]  
            ind_gamma_P.append(gamma1freq)
            ind_gamma_R.append(gamma2freq)
        ind_gamma_P = np.sum(ind_gamma_P, axis = 1)    
        ind_gamma_R = np.sum(ind_gamma_R, axis = 1)
        tstat, pval = stats.ttest_rel(ind_gamma_P, ind_gamma_R) 
        col_gamma_ttest.append(pval)
        col_gamma_P.append(np.median(ind_gamma_P))
        col_gamma_R.append(np.median(ind_gamma_R)) 
        
    
    #Lists to arrays
    psd_P = np.asarray(psd_P)
    psd_R = np.asarray(psd_R)
    
    col_alpha_P = np.asarray(col_alpha_P)
    col_alpha_R = np.asarray(col_alpha_R)
    
    col_beta_P = np.asarray(col_beta_P)
    col_beta_R = np.asarray(col_beta_R)
    
    col_gamma_P = np.asarray(col_gamma_P)
    col_gamma_R = np.asarray(col_gamma_R)
    
#####NSI Calculations & Plots by Layer
    #Create Sub-folder for large number of plots 
    if not os.path.exists(path + '/Analysis/PSD'):
        os.mkdir(path + '/Analysis/PSD')
    
    #Alpha
    ratio = (col_alpha_P - col_alpha_R) / (col_alpha_R + col_alpha_P) 
    alphaRatio_col = ratio 
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio': ratio, 'compLayer': compLayer})
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Alpha_SG_Collapsed_'+name+ '_'+stim+'.png', stim, 'SG')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Alpha_G_Collapsed_' +name+ '_'+stim+'.png', stim, 'G')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Alpha_IGU_Collapsed_'+name+ '_'+stim+'.png', stim, 'IGU')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Alpha_IGL_Collapsed_'+name+ '_'+stim+'.png', stim, 'IGL')
    
    #Beta
    ratio = (col_beta_P - col_beta_R) / (col_beta_R + col_beta_P) 
    betaRatio_col = ratio 
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio': ratio, 'compLayer': compLayer})
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Beta_SG_Collapsed_' +name+ '_'+stim+'.png', stim, 'SG')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Beta_G_Collapsed_'+name+ '_'+stim+'.png', stim, 'G')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Beta_IGU_Collapsed_' +name+ '_'+stim+'.png', stim, 'IGU')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Beta_IGL_Collapsed_'+name+ '_'+stim+'.png', stim, 'IGL')
    
    #Gamma
    ratio = (col_gamma_P - col_gamma_R) / (col_gamma_R + col_gamma_P) 
    gammaRatio_col = ratio 
    compLayer = laminar_labelTolayer(df1['layer'])
    df = pd.DataFrame({'ratio': ratio, 'compLayer': compLayer})
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Gamma_SG_Collapsed_' +name+ '_'+stim+'.png', stim, 'SG')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Gamma_G_Collapsed_' +name+ '_'+stim+'.png', stim, 'G')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Gamma_IGU_Collapsed_'+name+ '_'+stim+'.png', stim, 'IGU')
    plot_individual_nsi(df, path +'/Analysis/PSD/Layer_Gamma_IGL_Collapsed_' +name+ '_'+stim+'.png', stim, 'IGL')
    
#####NSI Heatmap Plot
    heatmapData = pd.DataFrame({'compLayer': compLayer, 
                           'alpha_col': alphaRatio_col, 'beta_col': betaRatio_col, 'gamma_col':  gammaRatio_col,\
                               'alpha_P':col_alpha_P,  'alpha_R':col_alpha_R, \
                               'beta_P':col_beta_P,  'beta_R':col_beta_R, \
                               'gamma_P':col_gamma_P,  'gamma_R':col_gamma_R})
    
    heatmapData_toPlot_mean = ([[heatmapData[heatmapData['compLayer']=='SG']['alpha_col'].mean(), 
                            heatmapData[heatmapData['compLayer']=='SG']['beta_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='SG']['gamma_col'].mean()],
                           [heatmapData[heatmapData['compLayer']=='G']['alpha_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='G']['beta_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='G']['gamma_col'].mean()],
                           [heatmapData[heatmapData['compLayer']=='IGU']['alpha_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='IGU']['beta_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='IGU']['gamma_col'].mean()],
                           [heatmapData[heatmapData['compLayer']=='IGL']['alpha_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='IGL']['beta_col'].mean(),
                            heatmapData[heatmapData['compLayer']=='IGL']['gamma_col'].mean()]])
    
    plot_laminar_profile_PSD(heatmapData_toPlot_mean,path +'/Analysis/PSD/PSD_ColumnarStack_'\
                                 +name+ '_'+stim+'.png', stim)
    
    heatmapData_all = ([[heatmapData[heatmapData['compLayer']=='SG']['alpha_col'], 
                            heatmapData[heatmapData['compLayer']=='SG']['beta_col'],
                            heatmapData[heatmapData['compLayer']=='SG']['gamma_col']],
                           [heatmapData[heatmapData['compLayer']=='G']['alpha_col'],
                            heatmapData[heatmapData['compLayer']=='G']['beta_col'],
                            heatmapData[heatmapData['compLayer']=='G']['gamma_col']],
                           [heatmapData[heatmapData['compLayer']=='IGU']['alpha_col'],
                            heatmapData[heatmapData['compLayer']=='IGU']['beta_col'],
                            heatmapData[heatmapData['compLayer']=='IGU']['gamma_col']],
                           [heatmapData[heatmapData['compLayer']=='IGL']['alpha_col'],
                            heatmapData[heatmapData['compLayer']=='IGL']['beta_col'],
                            heatmapData[heatmapData['compLayer']=='IGL']['gamma_col']]])
    
#####Plot PSDs by layer
    compLayer = laminar_labelTolayer(layer)
    
    #Supragranular Layer 
    sg = np.asarray(np.where(compLayer == 'SG')).flatten()
    SG_P = np.take(psd_P, sg, axis = 0)
    SG_R = np.take(psd_R, sg, axis = 0)
    plot_psd_band(SG_P[:,0:2], SG_R[:,0:2], path +'/Analysis/PSD/', stim, name, 'SG', 'Alpha')
    plot_psd_band(SG_P[:,2:6], SG_R[:,2:6], path +'/Analysis/PSD/', stim, name, 'SG', 'Beta')
    plot_psd_band(SG_P[:,6:14], SG_R[:,6:14], path +'/Analysis/PSD/', stim, name, 'SG', 'Gamma')
    
    #Granular Layer 
    g = np.asarray(np.where(compLayer == 'G')).flatten()
    G_P = np.take(psd_P, g, axis = 0)
    G_R = np.take(psd_R, g, axis = 0)
    plot_psd_band(G_P[:,0:2], G_R[:,0:2], path +'/Analysis/PSD/', stim, name, 'G', 'Alpha')
    plot_psd_band(G_P[:,2:6], G_R[:,2:6], path +'/Analysis/PSD/', stim, name, 'G', 'Beta')
    plot_psd_band(G_P[:,6:14], G_R[:,6:14], path +'/Analysis/PSD/', stim, name, 'G', 'Gamma')
    
    #Infragranular Upper Layer 
    igu = np.asarray(np.where(compLayer == 'IGU')).flatten()
    IGU_P = np.take(psd_P, igu, axis = 0)
    IGU_R = np.take(psd_R, igu, axis = 0)
    plot_psd_band(IGU_P[:,0:2], IGU_R[:,0:2], path +'/Analysis/PSD/', stim, name, 'IGU', 'Alpha')
    plot_psd_band(IGU_P[:,2:6], IGU_R[:,2:6], path +'/Analysis/PSD/', stim, name, 'IGU', 'Beta')
    plot_psd_band(IGU_P[:,6:14], IGU_R[:,6:14], path +'/Analysis/PSD/', stim, name, 'IGU', 'Gamma')
    
    #Infragranular Upper Layer 
    igl = np.asarray(np.where(compLayer == 'IGL')).flatten()
    IGL_P = np.take(psd_P, igl, axis = 0)
    IGL_R = np.take(psd_R, igl, axis = 0)
    plot_psd_band(IGL_P[:,0:2], IGL_R[:,0:2], path +'/Analysis/PSD/', stim, name, 'IGL', 'Alpha')
    plot_psd_band(IGL_P[:,2:6], IGL_R[:,2:6], path +'/Analysis/PSD/', stim, name, 'IGL', 'Beta')
    plot_psd_band(IGL_P[:,6:14], IGL_R[:,6:14], path +'/Analysis/PSD/', stim, name, 'IGL', 'Gamma')
    
    #Save data to file 
    pickle.dump({'heatmap': heatmapData_toPlot_mean, 'heatmap_allData': heatmapData_all}, 
    open(path + '/Data/FinalPSD_' + name + '_' + stim + '.pickle', 'wb'), protocol =-1)
        
####Wilcoxon tests

    #Holders 
    pVals = []
    pVals_rel = []
    pVals_rel_Wx = [] 
    labels = []
    median_P = []
    median_R = []
    conf_high = []
    conf_low = []
    
    #SG
    data1 = heatmapData[heatmapData['compLayer']=='SG']['alpha_P']
    data2 = heatmapData[heatmapData['compLayer']=='SG']['alpha_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('SG_Alpha')
    
    data1 = heatmapData[heatmapData['compLayer']=='SG']['beta_P']
    data2 = heatmapData[heatmapData['compLayer']=='SG']['beta_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('SG_Beta')
    
    data1 = heatmapData[heatmapData['compLayer']=='SG']['gamma_P']
    data2 = heatmapData[heatmapData['compLayer']=='SG']['gamma_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('SG_Gamma')
    
    #G
    data1 = heatmapData[heatmapData['compLayer']=='G']['alpha_P']
    data2 = heatmapData[heatmapData['compLayer']=='G']['alpha_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('G_Alpha')
    
    data1 = heatmapData[heatmapData['compLayer']=='G']['beta_P']
    data2 = heatmapData[heatmapData['compLayer']=='G']['beta_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('G_Beta')
    
    data1 = heatmapData[heatmapData['compLayer']=='G']['gamma_P']
    data2 = heatmapData[heatmapData['compLayer']=='G']['gamma_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('G_Gamma')
    
    #IGU
    data1 = heatmapData[heatmapData['compLayer']=='IGU']['alpha_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGU']['alpha_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGU_Alpha')
    
    data1 = heatmapData[heatmapData['compLayer']=='IGU']['beta_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGU']['beta_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGU_Beta')
    
    data1 = heatmapData[heatmapData['compLayer']=='IGU']['gamma_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGU']['gamma_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGU_Gamma')
    
    #IGL
    data1 = heatmapData[heatmapData['compLayer']=='IGL']['alpha_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGL']['alpha_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGL_Alpha')
    
    data1 = heatmapData[heatmapData['compLayer']=='IGL']['beta_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGL']['beta_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGL_Beta')
    
    data1 = heatmapData[heatmapData['compLayer']=='IGL']['gamma_P']
    data2 = heatmapData[heatmapData['compLayer']=='IGL']['gamma_R']
    stat, pVal = stats.wilcoxon(data1, data2)
    pVals_rel_Wx.append(pVal)
    ci = non_param_paired_CI(data1 * 1e10, data2 * 1e10, .95)
    conf_low.append(ci[0] / 1e10)
    conf_high.append(ci[1] / 1e10)
    median_P.append(np.median(data1))
    median_R.append(np.median(data2))
    labels.append('IGU_Gamma')

    
    #Write statistics to csv file
    datatofile = np.array([labels, pVals_rel_Wx, conf_high, conf_low, median_P, median_R])
    datatofile = datatofile.T
    headers = (['Layer', 'pVals_Wilxocon', 'conf_Low', 'conf_High', 'med_P','med_R'])
    pd.DataFrame(datatofile).to_csv(path +'/Analysis/PSD/LayerwisePSD_WilcoxonTestResults_' \
                                    + name + '_' + stim + '.csv', header = headers)
