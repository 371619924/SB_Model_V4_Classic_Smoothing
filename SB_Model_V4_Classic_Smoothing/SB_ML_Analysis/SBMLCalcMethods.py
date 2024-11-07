#!/opt/local/bin python

    #   SIMSEISCalcMethods.py    -   This code uses a Poisson distribution to generate state variable
    #       timeseries similar to those obtained by the various Nowcasting codes of Rundle et al.  
    #       The idea is to use a Poisson distribution, the common model used for seismicity interval statistics.
    #   
    #   Poisson distribution = 1 - exp(t/tau)
    #
    #   This code was written on a Mac using Macports python, but Anaconda python should work as well.
    
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2022 by John B Rundle, University of California, Davis, CA USA
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
    # documentation files (the     "Software"), to deal in the Software without restriction, including without 
    # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
    # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all copies or suSKLantial portions of the Software.
    # 
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
    # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    #   ---------------------------------------------------------------------------------------
    
import sys
import os
import numpy as np
from array import array
from numpy import random

import math

import numpy as np
import scipy

import matplotlib.pyplot as plt

from scipy.integrate import simps

    #################################################################
    #################################################################
    
    #   Calculate the basic time series
    
def coarse_grain_timeseries(index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries, number_time_steps):
    
    #   Function to coarse grain the various time series and to eventually produce a state variable time series
    
    total_time = time_plate[len(time_plate)-1]
    
    stress_list_raw     = [[] for i in range(number_time_steps)]    #   List of lists
    area_list_raw       = [[] for i in range(number_time_steps)]    #   List of lists
    magnitude_list_raw  = [[] for i in range(number_time_steps)]    #   List of lists
    
    stress_list         =   []
    time_list           =   []
    magnitude_list      =   []
    area_list           =   []
    
    delta_time = total_time/float(number_time_steps)
    
    for i in range(len(time_plate)):
        #   Find the coarse grain time index of the events
        index_event = (time_plate[i]/total_time)*number_time_steps  
        index_event = int(index_event-1)
        
#         print(i, index_event, stress_timeseries[i])

        stress_list_raw[index_event].append(stress_timeseries[i])
        
        
        magnitude_list_raw[index_event].append(magnitude_timeseries[i])
        area_list_raw[index_event].append(area_timeseries[i])
        
    for i in range(number_time_steps):
        if len(stress_list_raw[i]) > 0:
            try:
                mean_stress = np.mean(stress_list_raw[i])
            except:
                pass
            time_list.append((i+1)*delta_time)  #   Time mark is at the end of the time interval
            
            stress_list.append(mean_stress)
            magnitude_list.append(magnitude_list_raw[i])
            area_list.append(area_list_raw[i])
            
    delta_time = time_list[1] - time_list[0]
    
    state_var_list = []
    
    for i in range(len(magnitude_list)):
        number_events = len(magnitude_list[i])
        state_var = math.log10(1+number_events)
        state_var_list.append(state_var)
    
#     for i in range(len(magnitude_list)):
#         print(i, magnitude_list[i])
        
    return time_list, stress_list, magnitude_list, area_list, state_var_list
    
    #################################################################

    
def compute_ROC(times_months, state_variable, large_event_times, number_thresholds, TWindow, random_flag):

    #   First we find the min value, then progressively lower (actually raise) the threshold and determine the
    #       hit rate and false alarm rate
    
    true_positive               =   []
    false_positive              =   []
    true_negative               =   []
    false_negative              =   []
    
    threshold_value                 =   []
    
    min_value = min(state_variable)
#     min_value = 0.0
    max_value = max(state_variable)
    delta_threshold = (max_value - min_value) / float(number_thresholds-1)
    
    print('min_value: ',round(min_value,5))
    
    threshold = max_value + delta_threshold
#     threshold = max_value
    
    print()
    
    excluded_time = int(TWindow)
    
    print('number_thresholds', number_thresholds)
    print()
    
    for i in range(number_thresholds):
    
        threshold = threshold - delta_threshold
        fp = 0.
        tp = 0.
        tn = 0.
        fn = 0.
        
        if random_flag == False:
            percent_complete = round(100.0*(1. - threshold),2)
            print_string = 'Analyzed ' + str(percent_complete) + '% of ' + str(number_thresholds) + ' Threshold Values'
            print(print_string, end='\r', flush=True)
        
        for j in range(len(times_months) - excluded_time):  #   We exclude the last time that has incomplete data
        
            test_flag = True
            
            for k in range(len(large_event_times)):
            
                delta_time = large_event_times[k] - times_months[j]
                
                #   if value greater than threshold and at least 1 eq occurs within TWindow, tp
                
                if delta_time < TWindow and delta_time > 0 and float(state_variable[j]) > threshold and test_flag:
                    tp += 1.0
                    test_flag = False
                        
                #   if value greater than threshold, so predicted to occur within TWindow,
                #       but time to next large event is larger than TWindow, fp        
                
                if delta_time > TWindow and delta_time > 0 and float(state_variable[j]) > threshold and test_flag:
                    fp += 1.0
                    test_flag = False

                #   if value less than threshold, so predicted NOT to occur within TWindow, 
                #      and time to next large event is smaller than TWindow, fn   
                   
                if delta_time < TWindow and delta_time > 0 and float(state_variable[j]) < threshold and test_flag:
                    fn += 1.0
                    test_flag = False

                #   if value less than threshold so predicted NOT to occur within TWindow,
                #      and time to next large event is larger than TWindow,  tn      
                if delta_time > TWindow and delta_time > 0 and float(state_variable[j]) < threshold and test_flag:
                    tn += 1.0
                    test_flag = False
                
        true_positive.append(tp)
        false_positive.append(fp)
        true_negative.append(tn)
        false_negative.append(fn)
        
        threshold_value.append(threshold)
        
    print()
        
    return  true_positive, false_positive, true_negative, false_negative, threshold_value
    
    ######################################################################

def compute_ROC_rates(true_positive, false_positive, true_negative, false_negative):

    true_positive_rate      =   []
    false_positive_rate     =   []
    true_negative_rate      =   []
    false_negative_rate     =   []

    for i in range(len(true_positive)):
    
        tp = true_positive[i]
        fp = false_positive[i]
        tn = true_negative[i]
        fn = false_negative[i]
        
        if float(tp + fn) > 0:
            tpr = tp/(tp + fn)
            tpr = round(tpr,4)
            true_positive_rate.append(tpr)
            
        if float(fp + tn) > 0.:
            fpr = fp/(fp + tn)
            fpr = round(fpr,4)
            false_positive_rate.append(fpr)
            
        if float(fn + tp ) > 0.:
            fnr = fn/(fn + tp)
            fnr = round(fnr,4)
            false_negative_rate.append(fnr)
            
        if float(tn + fp) > 0.:
            tnr = tn/(tn + fp)
            tnr = round(tnr,4)
            true_negative_rate.append(tnr)
            
#     print(true_positive_rate)
#     print()
#     print(false_positive_rate)
            
    return true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate
    
    ######################################################################
    
# def calc_ROC_skill(values_window, times_window, true_positive, false_positive, true_negative, false_negative, \
#                     threshold_value, forecast_interval, mag_large, min_mag, plot_start_year,\
#                     data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
#                     Grid, Location, NSteps, delta_time_interval, lambda_mult, min_rate):
#                     
# # 
# #   ------------------------------------------------------------
# #
# #   Plot ROC and random ROCs
# 
#     true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
#                 SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
# 
# 
#     skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
# 
#         
#     return skill_score

    ##############################################r########################
    
def random_timeseries(times_window, values_window):

    random_values = []
    
    for i in range(len(values_window)):
        random_values.append(random.choice(values_window))

    return random_values
    
    ######################################################################
    
def calc_ROC_information_entropy(number_thresholds, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate):

    info_tp  =   0.
    
    print(true_positive_rate)
    
    for i in range(1,number_thresholds):
        diff = true_positive_rate[i] - true_positive_rate[i-1]
        
#         print('i, true_positive_rate[i], diff: ', i, true_positive_rate[i], diff)
        
        tp_term = 0.
        
        try:
            tp_term = diff * math.log2(diff)
        except:
            pass
            
        info_tp -= tp_term
        
    info_random = math.log2(float(number_thresholds)-1.)  

    return info_tp, info_random
    
    ######################################################################
    
def jensen_shannon_divergence(true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate):
            
    random_flag = False
    
    number_thresholds = len(true_positive_rate)
    
    uniprob = 1./float(number_thresholds)
    
    tp_prob         =   []
    random_prob     =   [] 
    
    for i in range(1,number_thresholds):
        diff = true_positive_rate[i] - true_positive_rate[i-1]
        
        tp_term = 0.
        try:
            tp_term = diff
        except:
            pass
            
        tp_prob.append(tp_term)
        random_prob.append(uniprob)
        
    m_prob = [0.5*(tp_prob[i] + random_prob[i]) for i in range(len(tp_prob))]
    
    kl_divergence_tp        = kullback_leibler(tp_prob, m_prob)
    kl_divergence_random    = kullback_leibler(random_prob, m_prob)
    
    kl_divergence      = kullback_leibler(tp_prob, random_prob)
    
    js_divergence = 0.5*(kl_divergence_tp + kl_divergence_random)
        
    return  js_divergence, kl_divergence
    
    ######################################################################
    
def kullback_leibler(prob1, prob2):

    kl_divergence = 0.
    for i in range(len(prob1)):
        sum_arg = 0.
        try:
            sum_arg = prob1[i] * math.log2(prob1[i]/prob2[i])
        except:
            pass
        kl_divergence += sum_arg

    return kl_divergence
    
    ######################################################################
    
