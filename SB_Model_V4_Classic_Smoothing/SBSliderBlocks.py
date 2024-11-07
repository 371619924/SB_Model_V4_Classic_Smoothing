#!/opt/local/bin python

    #
    #   Present code origin date: June 2, 2021.  
    #
    #   Driver code for the slider block model of an earthquake fault or driven threshold system.
    #
    #   SB Model of an earthquake fault.  This is an offshoot of the Rundle-Jackson (1977) model
    #       The RJ model is formulated in displacement variables.  The SB model is formulated in
    #       terms of stress variables.
    #
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2021 by John B Rundle, University of California, Davis, CA USA
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
    # documentation files (the     "Software"), to deal in the Software without restriction, including without 
    # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
    # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
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

import datetime
import dateutil.parser

import time
from time import sleep  #   Added a pause of 30 seconds between downloads

import math
import random
from random import random

from matplotlib import pyplot
import matplotlib.pyplot as plt

#from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

import SBCalcMethods
import SBPlotMethods

    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                  DEFINE FUNCTIONS

def read_data_file():

    #   Read in the data. This file was written by the generate_time_series function
    #   
    
    input_file = open("SB_Timeseries.txt", "r")
    
    index_plate = []
    time_plate =   []
    stress_timeseries   =   []
    area_timeseries     =   []
    magnitude_timeseries=   []
    
    i = -1
    for line in input_file:
        items = line.strip().split()
#         print(items)
        i += 1

        area = float(items[2])
        if area > 0.001:
            index_plate.append(int(float(items[0])))
            
            time_plate.append(float(items[1]))
            area_timeseries.append(math.log(area,10))
            
            stress_timeseries.append(float(items[3]))
            
            magnitude_timeseries.append(float(items[4]))
            
    input_file.close()
    
    return index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries
    
def change_time_interval(index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries):

    low_cutoff  = 0
    high_cutoff =   len(time_plate) - 1
    
    min_time = low_cutoff
    max_time = high_cutoff

    print()
    print('Current number of time steps is: ', len(time_plate))
    print()
    print('Maximum time is: ', time_plate[len(time_plate)-1]) 
    
    print()
    print('Change the minimum time for plotting? (y/n)')
    resp = input()
    if resp == 'y':
        min_time = float(input('Enter minimum time: '))
       
    print()
    print('Change the maximum time for plotting? (y/n)')
    resp = input()
    if resp == 'y':
        max_time = float(input('Enter maximum time: '))


    print('min_time, max_time: ', min_time, max_time)
    
    for i in range(len(time_plate)):

        if float(time_plate[i]) <= min_time:
            low_cutoff = i
            
        if float(time_plate[i]) <= max_time:
            high_cutoff = i

    low_cutoff = len(time_plate)-low_cutoff
    high_cutoff = len(time_plate)-high_cutoff
    
    if low_cutoff != high_cutoff:
        index_plate             = index_plate[-low_cutoff:-high_cutoff]
        time_plate              = time_plate[-low_cutoff:-high_cutoff]
        stress_timeseries       = stress_timeseries[-low_cutoff:-high_cutoff]
        area_timeseries         = area_timeseries[-low_cutoff:-high_cutoff]
        magnitude_timeseries    = magnitude_timeseries[-low_cutoff:-high_cutoff]
        
    else:
        print()
        print('Problem.  low_cutoff = high_cutoff.  This is not allowed.')
        print('No adjustments to time interval were made.')
        print()

    return index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries
    
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                           INPUTS

#   INPUTS:
            
if __name__ == '__main__':

    N = 100                                 #   Number of blocks = N * N
    time_steps = 100000                      #   Number of time steps
    transient_fraction = 0.05                #   Fraction of time_steps to neglect due to initial transient

    plate_rate = 0.0                        #   Added stress in plate update.  If 0, is zero velocity model
    max_cycles = 50                         #   Maximum number of cycles in the failure loop to avoid runaway
    
    smoothing_amplitude = 0.0005            #   Number << 1, controls the afterslip, smooths the stress field
    loss_fraction = 1.0                     #   If loss_fraction = 1.0, there is total stress transfer
    threshold_mean = 10.0                   #   Average value
    threshold_deviation = 0.0               #   Amplitude of the variation about the mean
                                            #   Note:  With random threshold, best to set threshold_deviation = 0
    DF = 2.5                                #   Fractal dimension of block threshold surface (if not random)
    min_area_fit = 0.0                      #   Lower limit on fit to FA data.  Equates to minimum value of Magnitude
    max_area_fit = 1.                       #   Upper limit on fit to FA data.  Equates to maximum value of Magnitude
    min_mag_fit = 0.5                        #   Lower limit on fit to FA data.  Equates to minimum value of Magnitude
    max_mag_fit = 2.5                       #   Upper limit on fit to FA data.  Equates to maximum value of Magnitude
    
    #   >>>>>>>>  NOTE!  For stability, we must have KC*( (2R)^2 - 1 ) < KL, assuming a uniform (constant) threshold
    
    R = 1                                   #   Assume this as default
    KL = 1.0                                #   Typically we will assume KL = 1
    KC = 0.1                                #   This can be adjusted
    
    threshold_type = 'Random'               #   Can be 'Fractal' or 'Random'
    
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                                       INPUT FLAGS
    
    generate_new_data               =   True
    
    plot_stress_timeseries_area     =   True
    plot_stress_timeseries_mag      =   True
    plot_frequency_area             =   True
    plot_frequency_mag              =   True

    print('Model:  Threshold: ' + threshold_type + '   Threshold Mean: ' + str(threshold_mean) + '   Threshold Deviation: ' +\
                str(threshold_deviation) + '   R: ' + str(R) + '  DF: ' + str(DF) + '  N: ' + str(N) + '  KL :' + str(KL) + \
                '   KC: ' + str(KC) )

    initial_block_threshold = SBCalcMethods.define_failure_threshold(threshold_type, threshold_deviation, threshold_mean, N, DF)
    
    ###################################################################################################################
    ###################################################################################################################
    #                                                                                               SPECIFY CALCULATIONS
    
    if generate_new_data:
    
        print('KC: ', KC)

        time_value, event_stress, block_stress_histories, magnitude_history = \
                SBCalcMethods.generate_time_histories(N, time_steps, plate_rate, \
                loss_fraction, R, initial_block_threshold, KL, KC, transient_fraction,\
                threshold_type, threshold_deviation, threshold_mean, DF, max_cycles, smoothing_amplitude)
                
    index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries = read_data_file()
    
    #   -------------------------------------------------------------
    
    print()
    print('Change the time interval for plotting? (y/n)')
    resp = input()
    if resp == 'y':
        index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries = change_time_interval(index_plate, time_plate, \
                stress_timeseries, area_timeseries, magnitude_timeseries)
                
    #################################################################
    
    if plot_stress_timeseries_area:
    
        SBPlotMethods.plot_timeseries_stress_area(index_plate, time_plate, stress_timeseries, area_timeseries, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            KL, KC, max_cycles, smoothing_amplitude)
            
    #################################################################
            
    if plot_stress_timeseries_mag:
    
        SBPlotMethods.plot_timeseries_stress_mag(index_plate, time_plate, stress_timeseries, magnitude_timeseries, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            KL, KC, max_cycles, smoothing_amplitude)
            
    #################################################################
            
    if plot_frequency_area:
    
        area_bins, log_freq_area_bins, bin_size = SBCalcMethods.freqAreaBins(area_timeseries)
    
        SBPlotMethods.plot_freqArea(area_timeseries, area_bins, log_freq_area_bins, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            min_area_fit, max_area_fit, KL, KC, max_cycles, smoothing_amplitude)
            
    #################################################################
            
    if plot_frequency_mag:
    
        mag_bins, log_freq_mag_bins, bin_size = SBCalcMethods.freqMagBins(magnitude_timeseries, min_mag_fit)
    
        SBPlotMethods.plot_freqMag(magnitude_timeseries, mag_bins, log_freq_mag_bins, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            min_mag_fit, max_mag_fit, KL, KC, max_cycles, smoothing_amplitude)
            
    #################################################################
    
