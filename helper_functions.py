##########################################################################################################
#Import required packages & functions
import numpy as np
import pandas as pd  
import operator
from scipy import signal 
from scipy import stats

##########################################################################################################
#Calculate corrected significane based upon Benjamini-Hochberg method

def benjamini_hochberg(p_vals, Q):
    
    #Prep sorted p_vals data 
    m = len(p_vals)              #Number of total tests
    indx = np.arange(m)          #Indices for original data
    indx_data = np.vstack((indx, p_vals)).T
    sorted_data = indx_data[indx_data[:,1].argsort()]
    
    #Benjamini Hocberg critical values 
    bh = (np.arange(m)+1) / m * Q 
    
    #Compare sorted p-vals to Benjamini Hochberg critical values 
    comp = sorted_data[:,1] < bh
    
    #Reorder comparisons to align with original input
    #results = np.vstack((indx, comp)).T
    results = np.array([sorted_data[:,0], comp, bh]).T
    resort = results[results[:,0].argsort()]
    
    return resort[:,1] , resort[:,2]

##########################################################################################################
#Calculate confidence intervals for Wilcoxon signed rank tests 

def non_param_paired_CI(sample1, sample2, conf):
  n = len(sample1)  
  alpha = 1-conf      
  N = stats.norm.ppf(1 - alpha/2) 

  # The confidence interval for the difference between the two population
  # medians is derived through the n(n+1)/2 possible averaged differences.
  diff_sample = sorted(list(map(operator.sub, sample2, sample1)))
  averages = sorted([(s1+s2)/2 for i, s1 in enumerate(diff_sample) for _, s2 in enumerate(diff_sample[i:])])

  # the Kth smallest to the Kth largest of the averaged differences then 
  # determine the confidence interval, where K is:
  k = np.math.ceil(n*(n+1)/4 - (N * (n*(n+1)*(2*n+1)/24)**0.5))

  CI = (round(averages[k-1],3), round(averages[len(averages)-k],3))
  return CI

##########################################################################################################
#Calculate non-smoothed firing rate per trial 
#Input: Matrix of threshold crossings
#Returns: 1D array containing the average firing rate per bin

def firing_rate_ms_bins_trial_nonsmoothed(data_x_ms):
    #Calculate time bins (50ms pre trial to 200ms post trial, binsize = 5ms)
    bins = np.arange(-50, 200, 5)

    #Bin data into corresponding time bins, sum activity within bins
    binned_data_x = data_x_ms.reshape((np.shape(data_x_ms)[0], len(bins), int(np.shape(data_x_ms)[1]/len(bins))))
    summed_binned_data_x =  np.sum(binned_data_x, axis = 2) 
        
    #Calculate firing rate according to bin size, dividing by the bin size in seconds 
    fr_byTrial = []
    for trial in np.arange(np.shape(summed_binned_data_x)[0]):
        binned_firing_rate = summed_binned_data_x[trial] / (5/1000) 
        
        fr_byTrial.append(binned_firing_rate)
        
    fr_byTrial = np.asarray(fr_byTrial)    
        
    return fr_byTrial    

##########################################################################################################
#Transition lainar label to laminar detph 
        
def laminar_labelTolayer(array):
    
    depth = [] 
    #Iterate through array
    for element in array: 
        if element == 'SG1':
            depth.append('SG')
        if element == 'SG2':
            depth.append('SG')
        if element == 'SG3':
            depth.append('SG')
        if element == 'G1':
            depth.append('G')
        if element == 'G2':
            depth.append('G')
        if element == 'IG1':
            depth.append('IGU')
        if element == 'IG2':
            depth.append('IGU')
        if element == 'IG3':
            depth.append('IGU')
        if element == 'IG4':
            depth.append('IGL')
        if element == 'IG5':
            depth.append('IGL')
        if element == 'IG6':
            depth.append('IGL')

    depth = np.asarray(depth)
    
    return depth 