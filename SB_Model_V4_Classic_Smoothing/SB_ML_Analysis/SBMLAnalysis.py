#!/opt/local/bin python

    #   SIMSEISTimeSeries.py    -   This code uses a Poisson distribution to generate state variable
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

import SBMLCalcMethods
import SBMLPlotMethods
import SBMLFileMethods

    #################################################################
    #################################################################
    
    #   PARAMETERS
    
    #   Elementary State Variable:     ESV = 1 - exp(t/Tau) 
    
    #   Three Primary Adjustable Parameters Are:
    #
    #       Tau:                Will be a random variable that determines the
    #                           between "large earthquakes"
    #   
    #       Threshold_SV:       Specifies the value of the state variable 
    #                           at which the "large earthquake" occurs. Can
    #                           be a random variable.
    #   
    #       Residual_SV:        Specifies the value of the state variable
    #                           when recovery begins (i.e., essentially the number
    #                           of aftershocks).  Can be a random variable.
    #
    #   Other Main Parameters:
    #
    #       TW, or forecast_interval:   In basic time units, which we assume
    #                           to be months
    #
    #   Total number of months:     NMonths
    
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                MAIN COD AND INPUTS    
if __name__ == '__main__':
    
    #  INPUTS
    
    Threshold_SV = 0.995
    sigma_threshold = 0.125
    sigma_threshold = 0.0
    
    Residual_SV = 0.0
    sigma_residual = 0.5
    sigma_residual = 0.0
    
    Tau = 25.0     #   California:  8 M>6.7 in 70 years, recurrence time 100 months
#     Tau = 50.0 
    
    TWindow_number_intervals = 6     #   3 year forecast time in lunar months
    
    number_thresholds = 50
    
    delta_mag_large = 0.7   #   Or whatever
    
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                       INPUT FLAGS
    
    Plot_Stress                     =   False
    Plot_State_Variable             =   False
    Plot_ROC_Stress                 =   True
    Plot_ROC_State                  =   True

    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                  DEFINE VARIABLES
    
    NMonths = 25000    #   For the statistics
    
    #   Assumes a zero velocity model and KL = 1.  Otherwise, we need to write a parameters file 
                
    index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries = SBMLFileMethods.read_data_file()
    
    total_time = time_plate[len(time_plate)-1]
    print('')
    print('Total Time: ', total_time)
    print('')
    
    number_time_steps = int(0.1 * (len(index_plate) + 1))
    
    #   Here I first find how much time has elapsed in total, and then take the desired number of time steps
    #       and combine the events into these coarse grained time steps, THEN assign magnitudes and stress to each time
    #       step.
#         
    time_list, stress_list, magnitude_list, area_list, state_var_list = SBMLCalcMethods.coarse_grain_timeseries\
        (index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries, number_time_steps)
        
    min_mag = min(magnitude_timeseries)
    max_mag = max(magnitude_timeseries)
    
    mag_large = min_mag + delta_mag_large*(max_mag - min_mag)
        
    delta_time = time_list[1] - time_list[0]
    TWindow = delta_time * TWindow_number_intervals
        
    large_event_times = []
    
    for i in range(len(time_list)):
        mag_flag = 0
        for j in range(len(magnitude_list[i])):
        
            if float(magnitude_list[i][j]) >= mag_large and mag_flag == 0:
                large_event_times.append(time_list[i])
                mag_flag = 1
                
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                               SPECIFY CALCULATIONS

    if len(large_event_times) == 0:
        print('No Large Events.  Decrease mag_large Variable')
        
    #################################################################
        
    if Plot_Stress:
    
        SBMLPlotMethods.plot_stress_timeseries(time_list, stress_list, random_flag)
        
    #################################################################
        
    if Plot_State_Variable:
    
        SBMLPlotMethods.plot_state_timeseries(time_list, state_var_list, random_flag)
        
    #################################################################
        
    if Plot_ROC_Stress:
    
        if len(large_event_times) > 0:
            SBMLPlotMethods.plot_temporal_ROC_stress(time_list, stress_list, large_event_times, \
                TWindow, number_thresholds)
                
    #################################################################
        
    if Plot_ROC_State:
    
        if len(large_event_times) > 0:
            SBMLPlotMethods.plot_temporal_ROC_state(time_list, state_var_list, large_event_times, \
                TWindow, number_thresholds)
        
    #################################################################
    
