#!/opt/local/bin python

    #   SIMSEISPlotMethods.py    -   This code uses a Poisson distribution to generate state variable
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
from matplotlib import gridspec


import SBMLCalcMethods

from numpy import trapz
    #################################################################
    #################################################################
    
def plot_timeseries(time_months, state_variable, activity, large_event_times, TauNom, TWindow):

    NMonths = int(40*TauNom)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    time_series      = []
    month_series     = []
    activity_series  = []
    
    for i in range(NMonths):
        time_series.append(state_variable[i])
        month_series.append(time_months[i])
        activity_series.append(activity[i])

    ax.plot(month_series, time_series, zorder=2, label='State Variable')
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(month_series))]
    ax.fill_between(month_series, min_plot_line, time_series, color='c', alpha=0.1, zorder=0)
    
    # tidy up the figure
    ax.grid(True, linestyle='--', lw=0.5, zorder=1)
#     ax.legend(loc='right')
    ax.set_title('Scaled Small Earthquake Quiescence (Simulated)')
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('State Variable (Scaled Quiescence)')
    
    ymin, ymax = ax.get_ylim()  
    plt.ylim(ymin,ymax)
    
    ax.plot(month_series, activity_series, color='b', linestyle='--', lw=0.65, zorder=2, label='Scaled Activity')
    
    for i in range(len(large_event_times)):
        if large_event_times[i] <= NMonths:
            xlarge = [large_event_times[i],large_event_times[i]]
            ylarge = [0.,0.2]
            ax.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3)
#         ax1.plot(xlarge[0],ylarge[1] , color='b', ms=8, marker='.', zorder=3)
        
    ax.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3, label='Large Events')
    
    ax.legend(loc='right')
    
    figure_name = './Data/Timeseries_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
#     
    plt.savefig(figure_name,dpi=300)

    return
    
    #################################################################
    
def plot_cumulative(recurrence_intervals, NMonths, TauNom, TWindow):

    n_bins = 100

    fig, ax = plt.subplots(figsize=(8, 5))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(recurrence_intervals, n_bins, density=True, histtype='step',
                           cumulative=True, color='b', label='Empirical Data', zorder=2)
                           
    Tau = np.mean(recurrence_intervals)

    y = [1. - math.exp(-(bins[i]/Tau)) for i in range(len(bins))]
    
#     print(y)
#     print(bins)
    
#     y = y.cumsum()
#     y /= y[-1]

    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical Poisson')
    
    bins = list(bins)
    ncount = list(n)
    ncount.append(1.0)
    
    xmin, xmax =plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(bins))]
    ax.fill_between(bins, min_plot_line, ncount, color='c', alpha=0.1, zorder=0)
    
    number_events = len(recurrence_intervals)

    # tidy up the figure
    ax.grid(True, linestyle='--', lw=0.5, zorder=1)
    ax.legend(loc='right')
    ax.set_title('Cumulative Distribution: ' + str(number_events) + ' Simulated Events', fontsize=16)
    ax.set_xlabel('Recurrence Intervals (Months)', fontsize=14)
    ax.set_ylabel('Likelihood of Occurrence', fontsize=14)
    
    figure_name = './Data/Cumulative_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
#     
    plt.savefig(figure_name,dpi=300)

    return
    
    #################################################################
    
def plot_histogram(recurrence_intervals, NMonths, TauNom, TWindow):

    fig, ax = plt.subplots(figsize=(8, 8))
    
    scale_factor = len(recurrence_intervals)
    
    count, bins, ignored = ax.hist(recurrence_intervals, 100, density = False, color='green', edgecolor='black', zorder=2, lw=0.4, label='Empirical Data') 
    
    number_events = len(recurrence_intervals)
    
    # tidy up the figure
    ax.grid(True, lw = 0.5, linestyle='--', zorder=0)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='right', fontsize=18)
    ax.set_title(str(number_events) + ' Simulated Events', fontsize=28)
    ax.set_xlabel('Recurrence Interval', fontsize=22)
    ax.set_ylabel('Number of Recurrence Intervals', fontsize=22)

    figure_name = './Data/Histogram_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
#     
    plt.savefig(figure_name,dpi=300)
    
    return
    
    #################################################################
    
def plot_temporal_ROC(time_months, state_variable, large_event_times, TWindow, Threshold_SV, TauNom):
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    show_data_box = True

    random_flag = False
    number_thresholds = 500
    number_thresholds = 200

    
    plot_random    =   True
    
    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SIMSEISCalcMethods.compute_ROC(time_months, state_variable, large_event_times, number_thresholds,\
            TWindow, Threshold_SV, random_flag)

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SIMSEISCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    info_tp, info_random = SIMSEISCalcMethods.calc_ROC_information_entropy\
                (number_thresholds, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    js_divergence, kl_divergence = SIMSEISCalcMethods.jensen_shannon_divergence\
                (true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    if len(true_positive_rate) != len(false_positive_rate):
        print()
        print('*****************************************************************')
        print()
        print('>>>>>> TWindow = ' + str(TWindow) +' too large, there are no false positives or true negatives.')
        print()
        print('*****************************************************************')
        print()
        return
        
    fig, ax = plt.subplots()

    label_text = 'ROC for Large Events' 

    ax.plot(false_positive_rate, true_positive_rate, linestyle='-', lw=1.0, color='r', zorder=3, label = label_text)
    
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    ax.minorticks_on()
    
    last_index = len(false_positive_rate)-1
    x_line = [false_positive_rate[0], false_positive_rate[last_index]]
    y_line = [true_positive_rate[0], true_positive_rate[last_index]]
    
    ax.plot(x_line, y_line, linestyle='-', lw=1.0, color='k', zorder=2, label = 'Random Mean')
    
    if plot_random:
    
        number_random_timeseries = 100
        number_random_timeseries = 20
    
        number_thresholds = 20
#         random_true_positive_rate_list = [[] for i in range(number_thresholds)]   #   +1 because we prepend 0. when we compute rates
    
        random_flag = True
    
        print()
    
        for i in range(number_random_timeseries):
    
            print_string = 'Random Timeseries ' + str(i+1) + ' of ' + str(number_random_timeseries) + ' Total Timeseries'
            print(print_string, end='\r', flush=True)
    
            random_values = SIMSEISCalcMethods.random_timeseries(time_months, state_variable)
        
            true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                    SIMSEISCalcMethods.compute_ROC(time_months, random_values, large_event_times, \
                    number_thresholds, TWindow, Threshold_SV, random_flag)
#                 
            true_positive_rate_random, false_positive_rate_random, false_negative_rate_random, true_negative_rate_random = \
                    SIMSEISCalcMethods.compute_ROC_rates\
                    (true_positive_random, false_positive_random, true_negative_random, false_negative_random)   
                    
#             print('len(random_true_positive_rate_list),len(true_positive_rate_random):', \
#                     len(random_true_positive_rate_list),len(true_positive_rate_random))

            if i == 0:
                random_true_positive_rate_list = [[] for i in range(len(true_positive_rate_random))]   
                    
            for j in range(len(true_positive_rate_random)):
                random_true_positive_rate_list[j].append(true_positive_rate_random[j])
#         
            ax.plot(false_positive_rate_random, true_positive_rate_random, linestyle='-', lw=2.0, color='cyan', zorder=1, alpha = 0.4)
# # 
# #   ------------------------------------------------------------
# #        
        stddev_curve    =   []
        for i in range(len(random_true_positive_rate_list)):
            stddev_curve.append(np.std(random_true_positive_rate_list[i]))
        
        random_upper = []
        random_lower = []
    
        for i in range(len(false_positive_rate_random)):
            random_upper.append(false_positive_rate_random[i] + stddev_curve[i])
            random_lower.append(false_positive_rate_random[i] - stddev_curve[i])
        
        ax.plot(false_positive_rate_random, random_upper, linestyle='dotted', lw=0.75, color='k', zorder=2, label = '1 $\sigma$ Confidence')
        ax.plot(false_positive_rate_random, random_lower, linestyle='dotted', lw=0.75, color='k', zorder=2)
         
# 
#   ------------------------------------------------------------
#
        skill_score_upper    =   trapz(random_upper,false_positive_rate_random)
        skill_score_lower    =   trapz(random_lower,false_positive_rate_random)
        stddev_skill_score = 0.5*(abs(skill_score_upper- 0.5) + abs(skill_score_lower - 0.5))
    
    print()
    print('--------------------------------------')
    print()
    print('Skill Score: ', round(skill_score,3))
    print()
    if plot_random:
        print('Skill Score Random: ', '0.5 +/- ' + str(round(stddev_skill_score,3) ))
        print()
    print('--------------------------------------')
    print()
# # 
#     ax.legend(bbox_to_anchor=(0, 1), loc ='right', fontsize=8)
    ax.legend(loc ='lower center', fontsize=8)
# # 
# #   ------------------------------------------------------------
# #
    relative_skill = abs(skill_score - 0.5)
    
    skill_index = - 100.0 * (relative_skill * math.log2(relative_skill) + (1.-relative_skill) * math.log2(1.0-relative_skill)  )
 
    textstr =       'Skill Score = ' + str(round(skill_score,3)) + \
                    '\nSkill Index = ' + str(round(skill_index,2)) + '%'\
                    '\n$I_{ROC}$ = ' + str(round(info_tp,2)) + ' Bits' +\
                    '\n$I_{Random}$ = ' + str(round(info_random, 2)) + ' Bits' +\
                    '\n$JSDiv$ = ' + str(round(js_divergence, 2)) + ' Bits' +\
                    '\n$KLDiv$ = ' + str(round(kl_divergence, 2)) + ' Bits' +\
                    '\n$T_W$ = ' + str(TWindow) +\
                    '\n' + r'$\tau$ = ' + str(round(TauNom,2)) +\
                    '\nThreshold = ' + str(Threshold_SV)
                    
    textstr =       'Skill Score = ' + str(round(skill_score,3)) + \
                    '\nSkill Index = ' + str(round(skill_index,2)) + '%'\
                    '\n$T_W$ = ' + str(TWindow) +\
                    '\n' + r'$\tau$ = ' + str(round(TauNom,2)) +\
                    '\nThreshold = ' + str(Threshold_SV)
# 
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    if show_data_box:
    # place a text box in bottom right in axes coords
        ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
        
    ax.grid(True, lw = 0.5, linestyle='--', zorder=0)
    
#     SupTitle_text = 'Receiver Operating Characteristic'
#     plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Receiver Operating Characteristic'
    plt.title(Title_text, fontsize=16)
    
#     Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
#     plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Hit Rate (TPR)', fontsize = 12)
    plt.xlabel('False Alarm Rate (FPR)', fontsize = 12)
    
    figure_name = './Data/Temporal_ROC_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NThr' + \
            str(number_thresholds) + '.png'
#     
    plt.savefig(figure_name,dpi=300)
#     plt.show()
        
    return

    ##############################################r########################
    
def plot_timeseries_precision(time_months, state_variable, activity, large_event_times, TauNom, TWindow, Threshold_SV):

    fig = plt.figure(figsize=(10, 6))        #   Define large figure and thermometer - 4 axes needed

    gs = gridspec.GridSpec(1,2,width_ratios=[15, 5], wspace = 0.2) 
    ax0 = plt.subplot(gs[0])

    startup_time = int(1000)
    NMonths = int(150*TauNom + startup_time)

#     fig, ax0 = plt.subplots(figsize=(8, 5))
    
    time_series      = []
    month_series     = []
    activity_series  = []
    
    for i in range(startup_time, NMonths):
        time_series.append(state_variable[i])
        month_series.append(time_months[i])
        activity_series.append(activity[i])
        
    ax0.plot(month_series, time_series, zorder=2, label='State Variable')
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(month_series))]
    ax0.fill_between(month_series, min_plot_line, time_series, color='c', alpha=0.1, zorder=0)
    
    # tidy up the figure
    ax0.grid(True, linestyle='--', lw=0.5, zorder=1)
#     ax.legend(loc='right')
    ax0.set_title('Simulated Quiescence', fontsize=14)
    ax0.set_xlabel('Time (Months)', fontsize = 14)
    ax0.set_ylabel('State Variable (Scaled Quiescence)', fontsize = 14)
    
    ymin, ymax = ax0.get_ylim()  
    plt.ylim(ymin,ymax)
    
    ax0.plot(month_series, activity_series, color='b', linestyle='--', lw=0.65, zorder=2, label='Scaled Activity')
    
    for i in range(len(large_event_times)):
        if large_event_times[i] >= startup_time and large_event_times[i] <= NMonths:
            xlarge = [large_event_times[i],large_event_times[i]]
            ylarge = [0.,0.2]
            ax0.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3)
#         ax1.plot(xlarge[0],ylarge[1] , color='b', ms=8, marker='.', zorder=3)
        
    ax0.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3, label='Large Events')
    
    ax0.legend(loc='right')
    
    #
    #   =======================================================================
    #
    #   Second plot: Precision    
    
    ax1 = plt.subplot(gs[1])
    frame1 = plt.gca()
    
    # 
    #   -------------------------------------------------------------
    #
    #   Plot ROC and random ROCs
                
    #   Might have to create a list with the threshold values in it
    
    show_data_box = False
    
    threshold_reduced   =   []
    precision           =   []
    
    random_flag = False
    number_thresholds = 200
    
    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SIMSEISCalcMethods.compute_ROC(time_months, state_variable, large_event_times, number_thresholds,\
            TWindow, Threshold_SV, random_flag)
            
    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SIMSEISCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    info_tp, info_random = SIMSEISCalcMethods.calc_ROC_information_entropy\
                (number_thresholds, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    js_divergence, kl_divergence = SIMSEISCalcMethods.jensen_shannon_divergence\
                (true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
    
    #    ----------------------------------------------------
    
    for i in range(1,len(true_positive)):
        if true_positive[i] > 0. or false_positive[i] > 0.:
            numer = true_positive[i]
            denom = false_positive[i] + true_positive[i]
            threshold_reduced.append(threshold_value[i])
            precision.append(numer/denom)
            
    precision.insert(0,1.)
    threshold_reduced.insert(0,max(state_variable))
    
    label_text = 'Precision for Large Events'

    precision = [precision[i]*100.0 for i in range(len(precision))]
    
    ax1.plot(precision, threshold_reduced, linestyle='-', lw=1.25, color='m', zorder=3, label = label_text)
    
    ax1.grid(linestyle = 'dotted', linewidth=0.5)
    
    ax1.minorticks_on()
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    relative_skill = abs(skill_score - 0.5)
    
    skill_index = - 100.0 * (relative_skill * math.log2(relative_skill) + (1.-relative_skill) * math.log2(1.0-relative_skill)  )
    
    if show_data_box:
 
        textstr =       'Skill Score = ' + str(round(skill_score,3)) + \
                    '\nSkill Index = ' + str(round(skill_index,1)) + '%'\
                    '\n$I_{ROC}$ = ' + str(round(info_tp,2)) + ' Bits' +\
                    '\n$I_{Random}$ = ' + str(round(info_random, 2)) + ' Bits' +\
                    '\n$JSDiv$ = ' + str(round(js_divergence, 2)) + ' Bits' +\
                    '\n$KLDiv$ = ' + str(round(kl_divergence, 2)) + ' Bits' +\
                    '\n' + r'$\tau$ = ' + str(round(TauNom,2)) +\
                    '\nThreshold = ' + str(Threshold_SV)
# 
    # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    # place a text box in bottom right in axes coords
        ax1.text(0.975, 0.025, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
    # 
# 
#   -----------------------------------------------------------------
#
    Title_text = 'Chance of Large Event ' + '\nWithin ' + str(round(TWindow,1)) + ' Months (PPV)'
    ax1.set_title(Title_text, fontsize=14)
    
    ax1.set_xlabel('Precision - PPV (%)', fontsize = 14)
    ax1.set_ylabel('State Variable (Scaled Quiescence)', fontsize = 14)
#     
#     # Save figure
    
    figure_name = './Data/Timeseries_Precision_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
#     
    plt.savefig(figure_name,dpi=300)

    return
    
    #################################################################
    
def plot_timeseries_accuracy(time_months, state_variable, activity, large_event_times, TauNom, TWindow, Threshold_SV):

    fig = plt.figure(figsize=(10, 6))        #   Define large figure and thermometer - 4 axes needed

    gs = gridspec.GridSpec(1,2,width_ratios=[15, 5], wspace = 0.2) 
    ax0 = plt.subplot(gs[0])

    NMonths = 2000

#     fig, ax0 = plt.subplots(figsize=(8, 5))
    
    time_series      = []
    month_series     = []
    activity_series  = []
    
    for i in range(NMonths):
        time_series.append(state_variable[i])
        month_series.append(time_months[i])
        activity_series.append(activity[i])
        
#     print(month_series)
#     print(time_series)

    
    ax0.plot(month_series, time_series, zorder=2, label='State Variable')
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(month_series))]
    ax0.fill_between(month_series, min_plot_line, time_series, color='c', alpha=0.1, zorder=0)
    
    # tidy up the figure
    ax0.grid(True, linestyle='--', lw=0.5, zorder=1)
#     ax.legend(loc='right')
    ax0.set_title('Scaled Small Earthquake Quiescence (Simulated)')
    ax0.set_xlabel('Time (Months)', fontsize = 12)
    ax0.set_ylabel('State Variable (Scaled Quiescence)', fontsize = 12)
    
    ymin, ymax = ax0.get_ylim()  
    plt.ylim(ymin,ymax)
    
    ax0.plot(month_series, activity_series, color='b', linestyle='--', lw=0.65, zorder=2, label='Scaled Activity')
    
    for i in range(len(large_event_times)):
        if large_event_times[i] <= NMonths:
            xlarge = [large_event_times[i],large_event_times[i]]
            ylarge = [0.,0.2]
            ax0.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3)
#         ax1.plot(xlarge[0],ylarge[1] , color='b', ms=8, marker='.', zorder=3)
        
    ax0.plot(xlarge, ylarge, color='r', linestyle='--', zorder=3, label='Large Events')
    
    ax0.legend(loc='right')
    
    #
    #   =======================================================================
    #
    #   Second plot: Precision    
    
    ax1 = plt.subplot(gs[1])
    frame1 = plt.gca()
    
    # 
    #   -------------------------------------------------------------
    #
    #   Plot ROC and random ROCs

                
    #   Might have to create a list with the threshold values in it
    
    threshold_reduced   =   []
    accuracy            =   []
    
    random_flag = False
    number_thresholds = 500
    
    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SIMSEISCalcMethods.compute_ROC(time_months, state_variable, large_event_times, number_thresholds,\
            TWindow, Threshold_SV, random_flag)
            
    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SIMSEISCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    info_tp, info_random = SIMSEISCalcMethods.calc_ROC_information_entropy\
                (number_thresholds,true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    js_divergence, kl_divergence = SIMSEISCalcMethods.jensen_shannon_divergence\
                (true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
    
    #    ----------------------------------------------------
    
    for i in range(1,len(true_positive)):
        if true_positive[i] > 0. or false_positive[i] > 0.:
            numer = true_positive[i] + true_negative[i]
            denom = true_negative[i] + true_positive[i] + false_positive[i] + false_negative[i]
            threshold_reduced.append(threshold_value[i])
            accuracy.append(numer/denom)
            
    accuracy.insert(0,1.)
    threshold_reduced.insert(0,max(state_variable))
    
    label_text = 'Accuracy for Large Events'

    accuracy = [accuracy[i]*100.0 for i in range(len(accuracy))]
    
    ax1.plot(accuracy, threshold_reduced, linestyle='-', lw=1.25, color='m', zorder=3, label = label_text)
    
    ax1.grid(linestyle = 'dotted', linewidth=0.5)
    
    ax1.minorticks_on()
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    relative_skill = abs(skill_score - 0.5)
    
    skill_index = - 100.0 * (relative_skill * math.log2(relative_skill) + (1.-relative_skill) * math.log2(1.0-relative_skill)  )
 
    textstr =       'Skill Score = ' + str(round(skill_score,3)) + \
                    '\nSkill Index = ' + str(round(skill_index,1)) + '%'\
                    '\n$I_{ROC}$ = ' + str(round(info_tp,2)) + ' Bits' +\
                    '\n$I_{Random}$ = ' + str(round(info_random, 2)) + ' Bits' +\
                    '\n$JSDiv$ = ' + str(round(js_divergence, 2)) + ' Bits' +\
                    '\n$KLDiv$ = ' + str(round(kl_divergence, 2)) + ' Bits' +\
                    '\n' + r'$\tau$ = ' + str(round(TauNom,2)) +\
                    '\nFailure Threshold = ' + str(Threshold_SV)
# 
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    # place a text box in bottom right in axes coords
    ax1.text(0.975, 0.025, textstr, transform=ax1.transAxes, fontsize=7,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
 
    # 
# 
#   -----------------------------------------------------------------
#
    Title_text = 'Accuracy of Nowcast ' + '\nWithin ' + str(round(TWindow,1)) + ' Months (PPV)'
    ax1.set_title(Title_text, fontsize=9)
    
    ax1.set_xlabel('Accuracy - ACC (%)', fontsize = 12)
    ax1.set_ylabel('State Variable (Scaled Quiescence)', fontsize = 12)
#     
#     # Save figure
    
    figure_name = './Data/Timeseries_Accuracy_Tau' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
#     
    plt.savefig(figure_name,dpi=300)

    return
    
    #################################################################
    
def plot_JS_Divergence(time_months, state_variable, large_event_times, number_thresholds, TWindow, Threshold_SV):

    #   This method plots the Jensen-Shannon divergence in bits for the precision relative to the Poisson probability

    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SIMSEISCalcMethods.compute_ROC(time_months, state_variable, large_event_times, number_thresholds,\
            TWindow, Threshold_SV, random_flag)
            
    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                    SIMSEISCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
    
    #    ----------------------------------------------------
    
    for i in range(1,len(true_positive)):
        if true_positive[i] > 0. or false_positive[i] > 0.:
            numer = true_positive[i]
            denom = false_positive[i] + true_positive[i]
            threshold_reduced.append(threshold_value[i])
            precision.append(numer/denom)
            
    precision.insert(0,1.)
    threshold_reduced.insert(0,max(state_variable))
    
    precision = [precision[i]*100.0 for i in range(len(precision))] #   Precision in  %
    
    return
    
    #################################################################
    
def plot_seismic_attractor(state_variable, TauNom, TWindow, NMonths):
        
#
#   ------------------------------------------------------------
#
        
    fig, ax = plt.subplots()
    
    state_variable_i            =   state_variable[:-1]
    state_variable_iplus1       =   state_variable[1:]
    
#     print(len(state_variable_i), state_variable_i)
#     print()    
#     print(len(state_variable_iplus1), state_variable_iplus1)
    
#     ax.plot(state_variable_i, state_variable_iplus1, linestyle='None', marker='.', markersize=3, zorder=3)
    ax.plot(state_variable_i, state_variable_iplus1, linestyle='-', lw= 0.35, marker='.', markersize=3, zorder=3)
    
    for i in range(1,len(state_variable_i)):
        thetan      = [state_variable[i-1]]
        thetanp1    = [state_variable[i]]
        ax.plot(thetan, thetanp1, linestyle='None', marker='.', markersize=8, zorder=3)
        
    min_theta = min(state_variable)
    max_theta = max(state_variable)
    
    ax.plot([min_theta, max_theta], [min_theta, max_theta], 'k--', lw=0.75, zorder=1)
    ax.plot([max_theta], [max_theta], marker='o', markersize=15, mfc='None', mec='k', mew=1.5, zorder=0)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
#     max_plot_line = [ymax for i in range(len(time_list_reduced))]
#     ax.fill_between(time_list_reduced , max_plot_line, log_number_reduced, color='c', alpha=0.1, zorder=0)
    
    ax.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'both')
    
#     ax.legend(loc = 'upper left', fontsize=7)
    
    #     
    #   ------------------------------------------------------------
    #
            
#     test_time_interval = delta_time_interval/0.07692
#     if abs(test_time_interval-1.0) <0.01:
#         str_time_interval = '1 Month'
#     elif abs(test_time_interval-0.25) < 0.01:
#         str_time_interval = '1 Week'
#     elif abs(test_time_interval-2.0) < 0.01:
#         str_time_interval = '2 Months'
#     elif abs(test_time_interval-3.0) < 0.01:
#         str_time_interval = '3 Months'
#         
#     Rmax_plot = str(round(max_rate,0))
#     if max_rate > 10000:
#         Rmax_plot = 'Inf.'
#         
#     textstr =   'EMA Samples (N): ' + str(NSteps) +\
#                 '\nTime Step: ' + str_time_interval +\
#                 '\n$R_{min}$: ' + str(round(min_rate,0)) +\
#                 '\n$R_{max}$: ' + Rmax_plot +\
#                 '\n$M_{min}$: ' + str(round(min_mag,2))
# 
# # 
# #     # these are matplotlib.patch.Patch properties
#  #    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)
# # 
# # #     # place a text box in upper left in axes coords
# #     ax.text(0.015, 0.02, textstr, transform=ax.transAxes, fontsize=7,
# #         verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8)
# 
# 
# #   ------------------------------------------------------------
# #
# 
#     ax.minorticks_on()
#     
#     ax.tick_params(axis='both', labelsize=9)
#     
#     delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
#     delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
#     
#     SupTitle_text = 'Attractor for Seismic State $\Theta(t)$ ' 
# 
#     plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
#     
#     Title_text = 'Within ' + str(round(delta_deg_lat,2)) + '$^o$ Latitude and ' + str(round(delta_deg_lng,2)) + '$^o$ Longitude of ' + Location\
#         + ' (Note Inverted Axes)'
#             
#     plt.title(Title_text, fontsize=9)
    
    plt.ylabel('$\Theta(t_{n+1})$', fontsize = 9)
    plt.xlabel('$\Theta(t_{n})$', fontsize = 9)
    
#     data_string_title = 'Attractor' + '_FI' + str(forecast_interval) + '_TTI' + str(test_time_interval) + \
#             '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lambda_mult) 
# 
    figure_name = './Data/SIMSEIS_Attractor_TauNom' + str(round(TauNom,2)) + '_TWindow' + str(TWindow) + '_NMonths' + str(NMonths) + '.png'
    plt.savefig(figure_name,dpi=300)
    
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################