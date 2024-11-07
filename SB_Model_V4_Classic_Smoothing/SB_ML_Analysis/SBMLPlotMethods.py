#!/opt/local/bin python

    #   SBMLPlotMethods.py    -   This code uses a Poisson distribution to generate state variable
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
    
def plot_stress_timeseries(time_list, stress_series):

    delta_time = time_list[1] - time_list[0]

    plt.plot(time_list, stress_series)
    
    Title_text = 'Stress vs. Time in Time Units of ' + str(round(delta_time,3))
    plt.title(Title_text, fontsize = 14)
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Stress', fontsize=14)
    
    figure_name = './Data/Stress_vs_Time.png'
    plt.savefig(figure_name, dpi=300)
    
#     plt.show()

    plt.close('all')

    return
    
    #################################################################
    
def plot_state_timeseries(time_list, state_var_list):

    delta_time = time_list[1] - time_list[0]
    
    plt.plot(time_list, state_var_list)
    
    plt.gca().invert_yaxis()
    
    Title_text = 'State vs. Time in Time Units of ' + str(round(delta_time,3))
    plt.title(Title_text, fontsize = 14)
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('State = $Log_{10}(1+Number(t))$', fontsize=14)
    
    figure_name = './Data/State_vs_Time.png'
    plt.savefig(figure_name, dpi=300)
    
    plt.close('all')
    
#     plt.show()

    return
    
    #################################################################

def plot_temporal_ROC_stress(time_list, stress_list, large_event_times, TWindow, number_thresholds):
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    show_data_box = True

    random_flag = False
    number_thresholds = 500
    number_thresholds = 50

    
    plot_random    =   True
    
    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SBMLCalcMethods.compute_ROC(time_list, stress_list, large_event_times, number_thresholds,\
            TWindow,random_flag)

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SBMLCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    info_tp, info_random = SBMLCalcMethods.calc_ROC_information_entropy\
                (number_thresholds, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    js_divergence, kl_divergence = SBMLCalcMethods.jensen_shannon_divergence\
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

    label_text = 'ROC for Large Events using Stress' 

    ax.plot(false_positive_rate, true_positive_rate, linestyle='-', lw=1.0, color='r', zorder=3, label = label_text)
    
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    ax.minorticks_on()
    
    last_index = len(false_positive_rate)-1
    x_line = [false_positive_rate[0], false_positive_rate[last_index]]
    y_line = [true_positive_rate[0], true_positive_rate[last_index]]
    
    ax.plot(x_line, y_line, linestyle='-', lw=1.0, color='k', zorder=2, label = 'Random Mean')
    
    if plot_random:
    
        number_random_timeseries = 100
        number_random_timeseries = 50
    
        number_thresholds = 20
#         random_true_positive_rate_list = [[] for i in range(number_thresholds)]   #   +1 because we prepend 0. when we compute rates
    
        random_flag = True
    
        print()
    
        for i in range(number_random_timeseries):
    
            print_string = 'Random Timeseries ' + str(i+1) + ' of ' + str(number_random_timeseries) + ' Total Timeseries'
            print(print_string, end='\r', flush=True)
    
            random_values = SBMLCalcMethods.random_timeseries(time_list, stress_list)
        
            true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                    SBMLCalcMethods.compute_ROC(time_list, random_values, large_event_times, \
                    number_thresholds, TWindow, random_flag)
#                 
            true_positive_rate_random, false_positive_rate_random, false_negative_rate_random, true_negative_rate_random = \
                    SBMLCalcMethods.compute_ROC_rates\
                    (true_positive_random, false_positive_random, true_negative_random, false_negative_random)   
                    
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
    print('Skill Score (Stress): ', round(skill_score,3))
    print()
    if plot_random:
        print('Skill Score Random: ', '0.5 +/- ' + str(round(stddev_skill_score,3) ))
        print()
    print('--------------------------------------')
    print()
# # 
#     ax.legend(bbox_to_anchor=(0, 1), loc ='right', fontsize=8)
    ax.legend(loc ='lower center', fontsize=6)
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
                    '\n$T_W$ = ' + str(TWindow) 
                    
    textstr =       'Skill Score (Stress) = ' + str(round(skill_score,3)) + \
                    '\nSkill Index (Stress) = ' + str(round(skill_index,2)) + '%'\
                    '\n$T_W$ = ' + str(round(TWindow,3))

# 
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    if show_data_box:
    # place a text box in bottom right in axes coords
        ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
        
    ax.grid(True, lw = 0.5, linestyle='--', zorder=0)
    
#     SupTitle_text = 'Receiver Operating Characteristic'
#     plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Receiver Operating Characteristic using Block Stress'
    plt.title(Title_text, fontsize=12)
    
#     Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
#     plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Hit Rate (TPR)', fontsize = 12)
    plt.xlabel('False Alarm Rate (FPR)', fontsize = 12)
    
    figure_name = './Data/Temporal_ROC_Stress' + '_TWindow' + str(TWindow) + '_NThresh' + \
            str(number_thresholds) + '.png'
#     
    plt.savefig(figure_name,dpi=300)
#     plt.show()
        
    return

    ##############################################r########################
    
def plot_temporal_ROC_state(time_list, state_var_list, large_event_times, TWindow, number_thresholds):
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    show_data_box = True

    random_flag = False
    number_thresholds = 500
    number_thresholds = 50

    
    plot_random    =   True
    
    true_positive, false_positive, true_negative, false_negative, threshold_value = \
            SBMLCalcMethods.compute_ROC(time_list, state_var_list, large_event_times, number_thresholds,\
            TWindow, random_flag)

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SBMLCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    info_tp, info_random = SBMLCalcMethods.calc_ROC_information_entropy\
                (number_thresholds, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate)
                
    js_divergence, kl_divergence = SBMLCalcMethods.jensen_shannon_divergence\
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

    label_text = 'ROC for Large Events using State' 

    ax.plot(false_positive_rate, true_positive_rate, linestyle='-', lw=1.0, color='r', zorder=3, label = label_text)
    
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    ax.minorticks_on()
    
    last_index = len(false_positive_rate)-1
    x_line = [false_positive_rate[0], false_positive_rate[last_index]]
    y_line = [true_positive_rate[0], true_positive_rate[last_index]]
    
    ax.plot(x_line, y_line, linestyle='-', lw=1.0, color='k', zorder=2, label = 'Random Mean')
    
    if plot_random:
    
        number_random_timeseries = 100
        number_random_timeseries = 50
    
        number_thresholds = 20
#         random_true_positive_rate_list = [[] for i in range(number_thresholds)]   #   +1 because we prepend 0. when we compute rates
    
        random_flag = True
    
        print()
    
        for i in range(number_random_timeseries):
    
            print_string = 'Random Timeseries ' + str(i+1) + ' of ' + str(number_random_timeseries) + ' Total Timeseries'
            print(print_string, end='\r', flush=True)
    
            random_values = SBMLCalcMethods.random_timeseries(time_list, state_var_list)
        
            true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                    SBMLCalcMethods.compute_ROC(time_list, random_values, large_event_times, \
                    number_thresholds, TWindow, random_flag)
#                 
            true_positive_rate_random, false_positive_rate_random, false_negative_rate_random, true_negative_rate_random = \
                    SBMLCalcMethods.compute_ROC_rates\
                    (true_positive_random, false_positive_random, true_negative_random, false_negative_random)   
                    
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
    print('Skill Score (State): ', round(skill_score,3))
    print()
    if plot_random:
        print('Skill Score Random: ', '0.5 +/- ' + str(round(stddev_skill_score,3) ))
        print()
    print('--------------------------------------')
    print()
# # 
#     ax.legend(bbox_to_anchor=(0, 1), loc ='right', fontsize=8)
    ax.legend(loc ='lower center', fontsize=6)
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
                    '\n$T_W$ = ' + str(TWindow) 
                    
    textstr =       'Skill Score (State) = ' + str(round(skill_score,3)) + \
                    '\nSkill Index (State) = ' + str(round(skill_index,2)) + '%'\
                    '\n$T_W$ = ' + str(round(TWindow,3))

# 
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    if show_data_box:
    # place a text box in bottom right in axes coords
        ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
        
    ax.grid(True, lw = 0.5, linestyle='--', zorder=0)
    
#     SupTitle_text = 'Receiver Operating Characteristic'
#     plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Receiver Operating Characteristic using Block State'
    plt.title(Title_text, fontsize=12)
    
#     Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
#     plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Hit Rate (TPR)', fontsize = 12)
    plt.xlabel('False Alarm Rate (FPR)', fontsize = 12)
    
    figure_name = './Data/Temporal_ROC_State' + '_TWindow' + str(TWindow) + '_NThresh' + \
            str(number_thresholds) + '.png'
#     
    plt.savefig(figure_name,dpi=300)
#     plt.show()
        
    return

    ##############################################r########################