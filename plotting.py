##########################################################################################################
# Import required Packages 
import numpy as np
import pandas as pd
import scipy.stats as ss
import pickle 
import matplotlib.mlab as mlab 
import matplotlib 
matplotlib.use('agg')
from matplotlib.pyplot import *
from statistics import * 
import seaborn as sns
from matplotlib.colors import  ListedColormap, LinearSegmentedColormap
from matplotlib import colors
import matplotlib.collections as collections
import cmasher as cmr

##########################################################################################################
def scatter2D_edgelabels_staticAxes(data1, data2, pvals, outputpath, modality, bh_corr):    
    
    #Prepare needed values
    ratios = (data1 - data2) / (data1 + data2)
    #significant = np.where(pvals < 0.05, 1, 0)
    significant = bh_corr
    labeledsignificant = np.multiply(ratios, significant)
    colorlabels = np.where(labeledsignificant > 0, 'r', np.where(labeledsignificant < 0, 'b', 'white'))
    edgelabels = 'k'     
    if modality == 'ERP': maxLim = np.int(np.ceil(np.max([np.nanmax(data1), np.nanmax(data2)]))+100)
    if modality == 'FR': maxLim = 1500   
    if modality == 'Control': maxLim = 2  
    if modality == 'ControlERP': maxLim = np.int(np.ceil(np.max([np.nanmax(data1), np.nanmax(data2)]))+100)

    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(4, 4))
    
    yscale('log')
    xscale('log')
    if modality == 'ERP':
        xlim(left = 50, right = maxLim)
        ylim(bottom = 50, top = maxLim)
        
    if modality == 'FR':
        xlim(left = 3, right = maxLim)
        ylim(bottom = 3, top = maxLim)
    
    if modality == 'Control':
        xlim(left = 0.01, right = maxLim)
        ylim(bottom = 0.01, top = maxLim)
        
    if modality == 'ControlERP':
        xlim(left = 10, right = maxLim)
        ylim(bottom = 10, top = maxLim)     
    
    xVals = np.arange(maxLim)
    
    scatter(data1, data2, s = 150, c=colorlabels, edgecolors = edgelabels)
    plot(xVals, xVals, color = 'k', linewidth = 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    savefig(outputpath, dpi = 500, format = 'png') 
    close(fig)
    clf()   
     
##########################################################################################################
def plot_compact_laminar_profile(dataframe, outputpath, stim):
    
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[100*mm,76*mm])
    order = ['SG','G','IGU','IGL']
    
    if stim == 'probe': color = 'darkviolet'
    if stim == 'context': color = 'grey'    
    sns_plot = sns.swarmplot(y = 'ratio', x = 'compLayer', data=dataframe, order = order,\
                            color = color, size = 4.5, edgecolor = 'gray', linewidth = 1, zorder = 4)
    
    if stim == 'probe': color = 'thistle'
    if stim == 'context': color = 'gainsboro'
    sns_plot = sns.boxplot(y = 'ratio', x = 'compLayer', data=dataframe, order = order, color = color,\
                           linewidth = 2, zorder = 3)
    sns_plot.axhline(0, linewidth = 1, color = 'grey', zorder=-1)
    sns_plot.set(xlabel = None)
    sns_plot.tick_params(length = 10, width = 2.5, bottom = False)
    
    matplotlib.pyplot.ylim(-0.5, 0.5)
    
    sns_plot.figure.savefig(outputpath, dpi = 300, format = 'png') 
    matplotlib.pyplot.clf()   
    
##########################################################################################################
def plot_fr_grandAvg(msData1, msData2, outputpath, shank, stim, layer):    

    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(4, 4))
    
    #Determine x-vals, mean firing rates, and standard errors
    x = np.arange(0, 50, 5)
    
    mean_data1 = np.mean(msData1, axis = 0)[10:20]
    mean_data2 = np.mean(msData2, axis = 0)[10:20]
    
    #Plots firing rate means 
    plot(x, mean_data2, 'b', linewidth = 3)    
    plot(x, mean_data1, 'r',linewidth = 3)
    
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpath + '/FR_' + str(shank) + '_' + stim + '_' + layer + '.png', \
            dpi = 300, format = 'png')
    close(fig)
    clf()    
    
##########################################################################################################
def plot_thalamus_ratios(dataframe, outputpath, stim):
    
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[25*mm,76*mm])
    
    if stim == 'probe': color = 'darkviolet'
    if stim == 'context': color = 'grey'    
    sns_plot = sns.swarmplot(y = 'ratio', x = 'label', data=dataframe,\
                            color = color, size = 4.5, edgecolor = 'gray', linewidth = 1, zorder = 4)
    
    if stim == 'probe': color = 'thistle'
    if stim == 'context': color = 'gainsboro'
    sns_plot = sns.boxplot(y = 'ratio', x = 'label', data=dataframe, color = color,\
                           linewidth = 2, zorder = 3)
    sns_plot.axhline(0, linewidth = 1, color = 'grey', zorder=-1)
    sns_plot.set(xlabel = None)
    sns_plot.tick_params(length = 10, width = 2.5, bottom = False)
    
    matplotlib.pyplot.ylim(-0.5, 0.5)
    
    sns_plot.figure.savefig(outputpath, dpi = 300, format = 'png') 
    matplotlib.pyplot.clf()       
    
##########################################################################################################
def plot_individual_nsi(dataframe, outputpath, stim, layer):
    
    #Initialize Figure
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[120*mm,200*mm])
    
    #order = ['SG','G','IGU','IGL']
    if layer == 'SG':
        data = dataframe[dataframe['compLayer']=='SG']
    if layer == 'G':
        data = dataframe[dataframe['compLayer']=='G']  
    if layer == 'IGU':
        data = dataframe[dataframe['compLayer']=='IGU'] 
    if layer == 'IGL':
        data = dataframe[dataframe['compLayer']=='IGL'] 
        
    
    if stim == 'probe': color = 'darkviolet'
    if stim == 'context': color = 'grey'    
    sns_plot = sns.swarmplot(y = 'ratio', data=data, \
                            color = color, size = 5, edgecolor = 'gray', linewidth = 1, zorder = 4)
    
    if stim == 'probe': color = 'thistle'
    if stim == 'context': color = 'gainsboro'
    sns_plot = sns.boxplot(y = 'ratio', data=data, color = color, linewidth = 2, zorder = 3)
    sns_plot.axhline(0, linewidth = 1, color = 'grey', zorder=-1)
    sns_plot.set(xlabel = None)

    sns_plot.tick_params(length = 15, width = 4, bottom = False, rotation = 90)  
    
    matplotlib.pyplot.ylim(-0.5, 0.5)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    sns_plot.figure.savefig(outputpath,dpi = 300, format = 'png') 
    matplotlib.pyplot.clf()   
    
##########################################################################################################
def plot_laminar_profile_PSD(data, outputpath, stim):   
    
    #Tick labels, when desired
    x_labels = ['a','b','g']
    y_labels = ['SG','G','IGU','IGL']
    
    #Customized color map 
    if stim == 'probe': p = [-0.1, -0.02, 0.02, 0.1]
    if stim == 'context': p = [-0.05, -0.01, 0.01, 0.05]
    f = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1])

    cmap = LinearSegmentedColormap.from_list('map_white', 
              list(zip(np.linspace(0,1), matplotlib.pyplot.cm.bwr(f(np.linspace(min(p), max(p)))))))
    
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(3, 6))
    #Initialize figure
    if stim == 'probe':
        sns.heatmap(data, vmin = -0.1, vmax = 0.1, center = 0, cbar = True, cmap = cmap, \
                    xticklabels=False, yticklabels=False, linewidths = 1, linecolor = 'silver')  
    if stim == 'context':
        sns.heatmap(data, vmin = -0.05, vmax = 0.05, center = 0, cbar = True, cmap = cmap, \
                    xticklabels=False, yticklabels=False, linewidths = 1, linecolor = 'silver')     
    
    savefig(outputpath, dpi = 500, format = 'png') 
    clf()   
    
##########################################################################################################      
def plot_psd_band(data1, data2, outputpath, stim, shank, layer, band):
    
    #Initialize figure
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[175*mm,125*mm])
    
    if band == 'Alpha':
        xaxisticks = ([0,1])
        xaxislabels = ([8,12])
        x = np.arange(0,2)
    if band == 'Beta': 
        xaxisticks = ([0,1,2,3])
        xaxislabels = ([16,20,24,28])  
        x = np.arange(0,4)
    if band == 'Gamma':
        xaxisticks = np.arange(0,8)
        xaxislabels = np.arange(32,64,4)
        x = np.arange(0,8)
    
    xticks(xaxisticks, xaxislabels)

    mean_data1 = np.mean(data1, axis = 0)
    mean_data2 = np.mean(data2, axis = 0)
    
    sem_data1 = ss.sem(data1, axis = 0)
    sem_data2 = ss.sem(data2, axis = 0)
    
    plot(x, mean_data2, 'b')
    fill_between(x, mean_data2 - sem_data2, mean_data2 + sem_data2, where = None, color = 'cornflowerblue', alpha = 0.3)
    
    plot(x, mean_data1, 'r')
    fill_between(x, mean_data1 - sem_data1, mean_data1 + sem_data1, where = None, color = 'lightcoral', alpha = 0.3)
    
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10, labelsize = 30)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5, labelsize = 30)
    
    savefig(outputpath + 'PSD_' + str(shank) + '_' + str(band)+ '_' + stim + '_' + layer + '.png', \
            dpi = 300, format = 'png')
    close(fig)
    clf()
    
##########################################################################################################
def plot_wavelet_diff_scales(t, freq, cwt, scales, outputpath):
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(5, 4))
    
    #Set no difference to white 
    divnorm=colors.TwoSlopeNorm(vmin=scales[0], vcenter=0, vmax=scales[1])
    
    im = matplotlib.pyplot.pcolormesh(t, freq, cwt, cmap = 'bwr', norm = divnorm)
    fig.colorbar(im)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    matplotlib.pyplot.savefig(outputpath, dpi = 500, format = 'png')
    clf()     
    
##########################################################################################################
def plot_evoked_channels_sem(channelData1, channelData2, outputpath):    
    
    #Initialize figure
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[90*mm,60*mm])
    
    x = np.linspace(-50,249,300)
    
    channelData1 = np.asarray(channelData1) * 1000000 #To positive uV
    channelData2 = np.asarray(channelData2) * 1000000 #To positive uV
    
    meanChl1 = np.nanmean(channelData1, axis = 0)
    meanChl2 = np.nanmean(channelData2, axis = 0)
    
    plot(x,meanChl1, 'r')
    plot(x,meanChl2, 'b')
     
    channelError1 = ss.sem(channelData1, axis = 0, nan_policy = 'omit')
    channelError2 = ss.sem(channelData2, axis = 0, nan_policy = 'omit')
    
    fill_between(x, meanChl1 - channelError1, meanChl1 + channelError1, color = 'lightcoral')
    fill_between(x, meanChl2 - channelError2, meanChl2 + channelError2, color = 'cornflowerblue')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    axvline(linewidth=1, color='grey')
    axhline(linewidth=1, color='grey')
    
    savefig(outputpath, dpi = 500, format = 'png')
    clf()    
        
##########################################################################################################
def plot_diff_sem(channelData1, channelData2, outputpath, stim, limits):    
    
    #Initialize figure
    mm = 1/25.4
    fig, ax = matplotlib.pyplot.subplots(1, figsize=[90*mm,60*mm])
    
    x = np.linspace(-50,249,300)
    
    channelData1 = np.asarray(channelData1) * 1000000 #To positive uV
    channelData2 = np.asarray(channelData2) * 1000000 #To positive uV
    diff = channelData1 - channelData2
    
    mean_diff = np.mean(diff, axis = 0)
    
    if stim == 'probe': color = 'darkviolet'
    if stim == 'context': color = 'grey' 

    plot(x,mean_diff, color = color)
     
    diffError = ss.sem(diff, axis = 0)
    
    if stim == 'probe': color = 'thistle'
    if stim == 'context': color = 'gainsboro'
    
    fill_between(x, mean_diff - diffError, mean_diff + diffError, color = color)
    
    if limits != 0:
        ax.set_ylim(limits)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    axvline(linewidth=1, color='grey')
    axhline(linewidth=1, color='grey')
    
    savefig(outputpath, dpi = 500, format = 'png')
    clf()         
    
##########################################################################################################
def plot_boxplot(data, dataTitles, figureTitle, figureName, outputpathFolder): 
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1,  figsize=(6, 3))
 
    boxplot(data, notch = False, vert = True, labels = dataTitles, showfliers = False)
    
    title(figureTitle)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpathFolder + '/' + figureName, dpi = 500, format = 'png')
    close(fig)
    clf()   
    
##########################################################################################################
def plot_laminar_profile_modulation_(alphaData, betaData, gammaData, outputpath):   
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1,  figsize=(6, 3))
    
    #Tick labels, when desired
    labels = ['SG','G','IGU','IGL']
    x = np.asarray([0,3,6,9])  # the label locations
    width = 0.7  # the width of the bars
    
    gamma = ax.bar(x - width, gammaData, width, color = 'white', edgecolor = 'r',linewidth = 3)
    beta = ax.bar(x, betaData, width,  color = 'white', edgecolor = 'r',linewidth = 3)
    alpha = ax.bar(x + width, alphaData, width, color = 'white', edgecolor = 'r',linewidth = 3)
    
    ax.set_ylim([0,0.7])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpath, dpi = 500, format = 'png') 
    clf()    
    
##########################################################################################################
def plot_boxplot_tight_double(data, limit, outputpath): 
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(3, 5))
    
    box3 = matplotlib.pyplot.boxplot(data[0:2], positions=[0.6, 1.2], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box3[item], color='red', linewidth = 3)
    for item in ['boxes']: 
        matplotlib.pyplot.setp(box3[item], color='red', linewidth = 3)    
    for item in ['medians']:    
        matplotlib.pyplot.setp(box3[item], color='k', linewidth = 3)
    for patch in box3['boxes']:
        patch.set(facecolor = 'white')
        
    box1 = matplotlib.pyplot.boxplot(data[0:2], positions=[0.6, 1.2], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box1[item], color='red', linewidth = 3)
    for item in ['boxes']: 
        matplotlib.pyplot.setp(box1[item], color='red', linestyle = ':', linewidth = 3)  
    for item in ['medians']:    
        matplotlib.pyplot.setp(box1[item], color='k', linewidth = 3)
    for patch in box1['boxes']:
        patch.set(facecolor = 'white')  
    
    box2 = matplotlib.pyplot.boxplot(data[2:4], positions=[2.2, 2.8], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['boxes', 'whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box2[item], color='blue', linewidth = 3)
    for item in ['medians']:    
        matplotlib.pyplot.setp(box2[item], color='k', linewidth = 3)
    for patch in box2['boxes']:
        patch.set(facecolor = 'white', linewidth = 3)  
                      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, limit])
    ax.axes.xaxis.set_visible(True)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpath, dpi = 500, format = 'png')
    close(fig)
    clf()        
    
##########################################################################################################
def plot_boxplot_tight_triple(data, limit, outputpath): 
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(3, 5))
    
    box3 = matplotlib.pyplot.boxplot(data[0], positions=[0.6, 1.2, 1.8], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box3[item], color='red', linewidth = 3)
    for item in ['boxes']: 
        matplotlib.pyplot.setp(box3[item], color='red', linewidth = 3)    
    for item in ['medians']:    
        matplotlib.pyplot.setp(box3[item], color='k', linewidth = 3)
    for patch in box3['boxes']:
        patch.set(facecolor = 'white')
        
    box1 = matplotlib.pyplot.boxplot(data[0], positions=[0.6, 1.2, 1.8], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box1[item], color='red', linewidth = 3)
    for item in ['boxes']: 
        matplotlib.pyplot.setp(box1[item], color='red', linestyle = ':', linewidth = 3)  
    for item in ['medians']:    
        matplotlib.pyplot.setp(box1[item], color='k', linewidth = 3)
    for patch in box1['boxes']:
        patch.set(facecolor = 'white')  
    
    box2 = matplotlib.pyplot.boxplot(data[1], positions=[2.8, 3.4, 4.0], notch = False, vert = True, \
                                     showfliers = False, widths = 0.5 ,patch_artist = True)
    for item in ['boxes', 'whiskers', 'fliers', 'caps']:
        matplotlib.pyplot.setp(box2[item], color='blue', linewidth = 3)
    for item in ['medians']:    
        matplotlib.pyplot.setp(box2[item], color='k', linewidth = 3)
    for patch in box2['boxes']:
        patch.set(facecolor = 'white', linewidth = 3)  
                      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, limit])
    ax.axes.xaxis.set_visible(True)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpath, dpi = 500, format = 'png')
    close(fig)
    clf()       
       
##########################################################################################################
def plot_laminar_profile_modulation_overlay(alphaDataP, betaDataP, gammaDataP,\
                                            alphaDataP_Corr, betaDataP_Corr, gammaDataP_Corr, \
                                            alphaDataR, betaDataR, gammaDataR, outputpath):   
    
    #Initialize figure
    fig, ax = matplotlib.pyplot.subplots(1,  figsize=(6, 3))
    
    #Tick labels, when desired
    labels = ['SG','G','IGU','IGL']
    x = np.asarray([0,3,6,9])  # the label locations
    width = 0.7  # the width of the bars
    
    #Pattern Data 
    gamma = ax.bar(x - width, gammaDataP, width, color = 'white', edgecolor = 'r',linewidth = 3, fill = False)
    beta = ax.bar(x, betaDataP, width,  color = 'white', edgecolor = 'r',linewidth = 3, fill = False)
    alpha = ax.bar(x + width, alphaDataP, width, color = 'white', edgecolor = 'r',linewidth = 3, fill = False)
    
    #Pattern Corrected Data 
    gamma = ax.bar(x - width,gammaDataP_Corr, width, color = 'white', edgecolor = 'r',linewidth = 3,alpha = 0.2, fill = False)
    beta = ax.bar(x, betaDataP_Corr, width,  color = 'white', edgecolor = 'r',linewidth = 3,alpha = 0.2, fill = False)
    alpha = ax.bar(x + width,alphaDataP_Corr, width, color = 'white', edgecolor = 'r',linewidth = 3,alpha = 0.2, fill = False)
    
    #Pattern Corrected Data 
    gamma = ax.bar(x - width,gammaDataP_Corr,width,color = 'white', edgecolor = 'r',linewidth = 3,linestyle=":", fill = False)
    beta = ax.bar(x, betaDataP_Corr, width,  color = 'white', edgecolor = 'r',linewidth = 3,linestyle=":", fill = False)
    alpha = ax.bar(x + width,alphaDataP_Corr,width,color = 'white', edgecolor = 'r',linewidth = 3,linestyle=":", fill = False)
    
    #Random Data first to introduce white hatch colors 
    gamma = ax.bar(x - width, gammaDataR, width, fill = False, edgecolor = 'b',linewidth = 3, alpha = 0.7)
    beta = ax.bar(x, betaDataR, width, fill = False, edgecolor = 'b',linewidth = 3, alpha = 0.7)
    alpha = ax.bar(x + width, alphaDataR, width, fill = False,  edgecolor = 'b',linewidth = 3, alpha = 0.7)
    
    ax.set_ylim([0,0.7])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.tick_params(axis='both', which='major', width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=2.5, length=5)
    
    savefig(outputpath, dpi = 500, format = 'png') 
    clf() 