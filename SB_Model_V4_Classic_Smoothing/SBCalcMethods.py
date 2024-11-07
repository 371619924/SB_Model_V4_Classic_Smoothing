#!/opt/local/bin python

    #   Present code origin date: June 2, 2021.  
    #
    #   SBCalcMethods.py  Methods for calculting data generated in the SB model.
    #
    #   SB Model of an earthquake fault.  This is an offshoot of the Rundle-Jackson (1977) model
    #       The RJ model is formulated in displacement variables.  The SB model is formulated in
    #       terms of stress variables.
    
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2020 by John B Rundle, University of California, Davis, CA USA
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

import scipy
from scipy import *
from scipy import fftpack

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

import SBPlotMethods


    #################################################################
    ################################################################

    #
    #   Set up the state list.  We assume an N x N system
    
def load_blocks(block_stress, block_threshold, plate_rate, N, KL, KC):

    #   Move the plate  -   We will use stress free boundary conditions
    
    #   -------------------------------------------------------------
    
    diff_stress = np.subtract(block_threshold,block_stress)
#     print('diff_stress.min(): ', diff_stress.min())
    
    if plate_rate < 0.00001:
    
        diff_stress = np.subtract(block_threshold,block_stress)
        
        plate_displ = diff_stress.min()/KL
        
        if plate_displ < 0.0:               #   In the event that there is small negative value
            plate_displ = 0.0   #   Last event probably did not finish in the cycles allowed
        
    else:
        plate_displ = plate_rate

    block_stress += KL*plate_displ

    #   -------------------------------------------------------------
    
    return block_stress, plate_displ
    
    ######################################################################
        
def find_failed_blocks(timei, block_stress, failed_blocks, block_threshold, initial_block_threshold, N):

    number_failed_blocks = 0
    
    for i in range(N):
        for j in range(N):
            if block_stress[i,j] >= block_threshold[i,j]:
                failed_blocks[i,j] += 1
                number_failed_blocks += 1

    return failed_blocks, number_failed_blocks, block_threshold
    
    ######################################################################
    
def update_block_state(initial_block_threshold, block_stress, block_displ, failed_blocks, \
        R, N, loss_fraction, number_neighbors, stress_transfer, K_total, KL, KC):

    #   In this version, stress is transferred to the Moore neighborhood around a failed block
    
    initial_block_stress = np.zeros((N,N))
    residual_block_stress= np.zeros((N,N))
    
    #   -------------------------------------------------------

    for i in range(N):
        for j in range(N):
            initial_block_stress[i,j] = block_stress[i,j]
            residual_block_stress[i,j] = (1.0 - loss_fraction) * block_stress[i,j]
            
    #   -------------------------------------------------------
    
    #   BLOCK FAILS:
    #   Any block that fails must stay at the residual stress for a time interval until it heals
    for i in range(N):
        for j in range(N):
            if failed_blocks[i,j] >= 1:            #   Reduce stress if a block fails or has failed

                block_displ[i,j] += (block_stress[i,j] - residual_block_stress[i,j])/K_total[i,j]
                block_stress[i,j] = residual_block_stress[i,j]
                    
    #   -------------------------------------------------------
    
#       TRANSFER STRESS:
#       To neighboring blocks if a block has failed.  If it has already failed, it must remain at 
#           the residual stress until it heals

    for i in range(N):
        for j in range(N):
        
#           Transfer stress only if the central block fails
            if failed_blocks[i,j]== 1:   
        
                for ki in range(i-R, i+R+1):          #   Blocks to above and below i,j
                    if ki >= 0 and ki < N:
#                     
                        for kj in range(j-R, j+R+1):      #   Blocks to left and right of i,j
                            if kj >= 0 and kj < N and (ki,kj) != (i,j):

                                if failed_blocks[i,j] > 0: 
                                    block_stress[ki,kj] += \
                                          stress_transfer[i,j]*(initial_block_stress[i,j] - residual_block_stress[i,j])
                                          
    return block_stress, block_displ
    
    ######################################################################
    
def generate_time_histories(N, time_steps, plate_rate, loss_fraction, R, initial_block_threshold, \
        KL, KC, transient_fraction,  threshold_type, threshold_deviation, threshold_mean, DF, max_cycles, smoothing_amplitude):
    
#   Set up the initial variables
    
#   -------------------------------------------------------
 
#      Fraction of stress that is lost, not re-distributed given by loss_fraction
#      Note that total stress transfer to all connected blocks cannot exceed 1

    transient_number = transient_fraction * time_steps
    
    print()
    print('Neglecting the first ' + str(transient_number) + ' events')
    print()

    N2 = N*N

    number_neighbors, stress_transfer, K_total = calc_neighbors(N,R,loss_fraction,KL,KC)

    block_stress_histories  = np.zeros((N2,time_steps)) #   Is not used
    magnitude_history       = np.zeros(time_steps)
    moment_history          = np.zeros(time_steps)
    
    time_value      =   []
    event_area_list =   []
    event_stress    =   []
    
    time_list               =   []
    plate_time_list         =   [] 
    event_area_list         =   [] 
    event_size_list         =   []
    mean_stress_list        =   []
    magnitude_list          =   []
    average_stress_list     =   []

    
#   -------------------------------------------------------

    #  Define initial state
    
    block_stress = np.zeros((N,N))
    
    #   Renormalize so that blocks fail right away
    
    number_neighbors, stress_transfer, K_total = calc_neighbors(N,R,loss_fraction,KL,KC)
    
    diff_stress = np.subtract(initial_block_threshold,block_stress)
        
    plate_displ = diff_stress.min()
    
    block_stress += plate_displ      #   So blocks should begin failing at the start - This actually works even though
                             
#   -------------------------------------------------------

    
    data_file = open('SB_Timeseries.txt',"w")
    data_file.close()
    
    max_failed_blocks = 0
    time_max = 0
    plate_time = 0.0
    total_failed_blocks = 0
    event_area = 0.
    
#   -------------------------------------------------------

    arg_plast_ratio_trunc = 'N/A'
    block_slip = 'N/A'
    
    #  Time Loop
    for i in range(time_steps):
    
        failed_blocks           = np.zeros((N,N))
        failed_blocks_area      = np.zeros((N,N))
        block_threshold         = np.zeros((N,N))
        block_displ             = np.zeros((N,N)) 
        initial_block_stress    = np.zeros((N,N))
        
        block_threshold = initial_block_threshold.copy()
        
        timei = i

        block_flags = np.zeros((N,N))    
        
        #   Move the plate, which updates the block stress
        block_stress, plate_displ = load_blocks(block_stress, block_threshold, plate_rate, N, KL, KC)
        
        for j in range(N):
            for k in range(N):
                initial_block_stress[j,k] = block_stress[j,k]
        
        plate_time += plate_displ
        
        time_value.append(plate_time)
        
        failed_blocks, number_failed_blocks, block_threshold = \
                find_failed_blocks(timei, block_stress, failed_blocks, block_threshold, block_threshold, N)
                
        ncycles = 0
        
        toplings = 0
        
        while number_failed_blocks > 0 and ncycles < max_cycles:   #   Last condition is to prevent the iteration going on forever
        
            ncycles += 1
        
            block_stress, block_displ = update_block_state(initial_block_threshold, block_stress, block_displ, \
                    failed_blocks, R, N, loss_fraction, number_neighbors, stress_transfer, K_total, KL, KC)
                    
            failed_blocks, number_failed_blocks, block_threshold = \
                    find_failed_blocks(timei, block_stress, failed_blocks, block_threshold, initial_block_threshold, N)
        
            for j in range(N):
                for k in range(N):
                    if failed_blocks[j,k] >= 1:
                        failed_blocks_area[j,k] = 1      #   True even if its already == 1
                        
#   -------------------------------------------------------

    #   If we hit the max_cycles limit, reduce the stress to the threshold value and adjust the displacement values
    #       to account for the reduced stress
    
        if ncycles == max_cycles:
    
            diff_stress = block_stress - initial_block_threshold
            max_diff_stress = np.max(diff_stress)
        
            block_stress -= max_diff_stress
        
            for ki in range(N):
                for kj in range(N):
                    block_displ[ki,kj] -= diff_stress[ki,kj] - block_displ[ki,kj]
        
#   -------------------------------------------------------

        #  Smooth the stress field

        if i >= transient_number:       #   Smooth the stress field with a logistic function in time
            
            #   Variable logistic_smoothing_amplitude is small for low stress, and higher for higher stress
            
            arg_plast_ratio = 0.999
            
            try:
                arg_plast_ratio = np.mean(block_stress)/np.mean(average_stress_list[-1500:])
            except:
                pass
                
            arg_plast_ratio_trunc = round(arg_plast_ratio, 4)
            
            delta_plast = 1./(arg_plast_ratio-1)
            
            try:
                exp_plast = math.exp(delta_plast)
            except:
                exp_plast = 1.e-10 * np.sign(delta_plast)
                
            logistic_smoothing_amplitude = smoothing_amplitude * exp_plast/(1. + exp_plast)
            
            new_block_stress = smooth_stress_field(block_stress, logistic_smoothing_amplitude, N, total_failed_blocks)

            mean_displ = 0.
            number_slipped_blocks = 0
            
            for k in range(N):
                for j in range(N):
                    block_displ[k,j] += (block_stress[k,j] - new_block_stress[k,j])/K_total[k,j]    # Aftership
                    block_stress[k,j] = new_block_stress[k,j]
                    if failed_blocks_area[j,k] > 0:
                        mean_displ += block_displ[j,k]
                        number_slipped_blocks += 1

            mean_displ = mean_displ/float(number_slipped_blocks)
            
            block_slip = round(mean_displ,4)
                    
#   -------------------------------------------------------
        
        event_area = 0
        for j in range(N):
            for k in range(N):
                event_area += failed_blocks_area[j,k]
                
        event_area_list.append(event_area)
        event_stress.append(np.mean(block_stress))
        
        total_failed_blocks += event_area
        
#   -------------------------------------------------------

    #   Compute moment history, magnitudes and other statistics
    
        if i >= transient_number:       #   Smooth the stress field with a logistic function in time
                        
            moment_history[i] = KL * mean_displ * event_area      #   Normal definition
        
            try:
                magnitude_history[i] = math.log(moment_history[i],10)
            except:
                pass
                    
            event_magnitude = magnitude_history[i]

            data_file = open('SB_Timeseries.txt',"a")
            print(timei, plate_time, event_area, np.mean(block_stress), event_magnitude, file=data_file)
            data_file.close()
            
#   -------------------------------------------------------
            
        kj=-1
        for j in range(1,N-1):
            for k in range(1,N-1):
            
                kj += 1
                block_stress_histories[kj,i] = block_stress[j,k]
                
        if event_area > max_failed_blocks and timei >= transient_number:
            max_failed_blocks = event_area
            time_max = timei
#             
        max_failed_blocks_print = str(int(max_failed_blocks))
        time_max_print = str(time_max)
        if timei < transient_number:
            max_failed_blocks_print = 'N/A'
            time_max_print = 'N/A'
            
#   -------------------------------------------------------
            
        #   Compute standard deviation of stress
        
        stress_list = []
        for ki in range(N):
            for kj in range(N):
                stress_list.append(block_stress[ki,kj])
        stress_std = np.std(stress_list)
        
        fraction_completed = 100.0 * float(i+1)/float(time_steps)
        print_string = '> Time: ' + str(i+1) + ' of '  + str(time_steps) + \
                    '   Max Area: ' + max_failed_blocks_print +  \
                '   At Time: ' + time_max_print + '  Completed: >' + str(int(fraction_completed)) + '%' +\
                '   Stress Ratio: ' + str(arg_plast_ratio_trunc) +\
                '   Current Area: ' + str(int(event_area)) +  '  Number Cycles: ' + str(ncycles) +\
                '  Mean Displ: ' + str(block_slip) + '    '
                
        print(print_string, end="\r", flush=True)
        
        average_stress = np.mean(block_stress)
        average_stress = round(average_stress,4)
        
        average_stress_list.append(average_stress)
        
        if i >= transient_number:
            time_list.append(timei) 
            plate_time_list.append(round(plate_time,4)) 
            event_area_list.append(event_area) 
            mean_stress_list.append(average_stress)
            magnitude_list.append(round(event_magnitude,3))

    #   -------------------------------------------------------------
    
    # Correct the magnitudes to give b = 1
    
    min_fit = 0.
    max_fit = 0.667*max(magnitude_list)
    
    mag_bins, log_freq_mag_bins, bin_size = freqMagBins(magnitude_list, min_fit)
    
    slope, cept, errs, errc, s2, kmax = fit_line_to_mag_data(magnitude_list, mag_bins, min_fit, max_fit)
#     
    #   -------------------------------------------------------------
    
    open('SB_Timeseries.txt', 'w').close()  #   Empty the text file prior to re-writing with corrected data
    
    #   Write the data file and adjust magnitudes so that b = 1
    
    data_file = open('SB_Timeseries.txt',"w")
    
    bval = - slope
    
    for i in range(len(magnitude_list)):
        magnitude_list[i] = bval*magnitude_list[i]
        
        print(time_list[i], plate_time_list[i], event_area_list[i], mean_stress_list[i], magnitude_list[i], file=data_file)
                
    data_file.close()

#         
#     #   -------------------------------------------------------------

    print()
                
    return time_value, event_stress, block_stress_histories, magnitude_history
    
    ######################################################################    
        
def define_failure_threshold(threshold_type, threshold_deviation, threshold_mean, N, DF):

    if threshold_type == 'Random':
    
        block_threshold = np.ones((N,N))    #   Start with constant threshold
    
        for i in range(N):      #   Phase transition as variance of threshold -> 0
            for j in range(N):
                delta = 2.0*(random()-0.5) * threshold_deviation    #
                block_threshold[i,j] += delta                       #   Threshold varies randomly 
                                                                #   between 1.0 +/- (threshold_deviation)
                                                                
    elif threshold_type == 'Fractal':
    
        block_threshold = genSurface(N,DF)
        block_threshold -= 0.5                              #   Default runs from 0. to 1.
        block_threshold *= threshold_deviation
        block_threshold += threshold_mean
        
        print('np.amin(block_threshold), np.amax(block_threshold)', np.amin(block_threshold), np.amax(block_threshold))
            
    return block_threshold
    
    ######################################################################

def genSurface(N,D):
    
        #   This is the spectral synthesis/filter method of constructing a fractal
        #   Adapted from:  http://shortrecipes.blogspot.com/2008/11/python-isotropic-fractal-surface.html
        
        H=1-(D-2)   #   Hausdorff dimension
        
        X=np.zeros((N,N),complex)
        A=np.zeros((N,N),complex)
        
        powerr=-(H+1.0)/2.0

        for i in range(int(N/2)+1):
                for j in range(int(N/2)+1):
                    phase=2*math.pi*np.random.rand()

                    if i != 0 or j != 0:
                        rad=(i*i+j*j)**powerr*np.random.normal()
                    else:
                        rad=0.0

                    A[i,j]=complex(rad*np.cos(phase),rad*np.sin(phase))

                    if i == 0:
                        i0=0
                    else:
                        i0=N-i
                    if j == 0:
                        j0=0
                    else:
                        j0=N-j

                    A[i0,j0]=complex(rad*np.cos(phase),-rad*np.sin(phase))


        A.imag[int(N/2)][0]=0.0
        A.imag[0,int(N/2)]=0.0
        A.imag[int(N/2)][int(N/2)]=0.0

        for i in range(1,int(N/2)):
                for j in range(1,int(N/2)):
                        phase=2*math.pi*np.random.rand()
                        rad=(i*i+j*j)**powerr*np.random.normal()
                        A[i,N-j]=complex(rad*np.cos(phase),rad*np.sin(phase))
                        A[N-i,j]=complex(rad*np.cos(phase),-rad*np.sin(phase))

        itemp=fftpack.ifft2(A)
        itemp=itemp-itemp.min()
        
        #   The plotted surface array in the original code is the real part of A scaled to, say, 15.0
        #   i.e., im=Aa.real/Aa.real.max()*15.0, and then im is plotted.  See the original code.
        
        complex_array = itemp

        fractal_surface = complex_array.real/complex_array.real.max()
        
        return fractal_surface
        
    ######################################################################
    
def freqAreaBins(area_timeseries):

    #   Note that the elements in the timeseries are Log_10(area)

    print_data = False
    
    # Create arrays of length i filled with zeros

    area =   np.zeros(len(area_timeseries))

    # Bins for Number-Area plot

    min_area = 0.0
    max_area = max(area_timeseries)

    bin_size= 0.025

    number_area_bins     =   (max_area - min_area) / bin_size + 5      #   Bin size = 0.1.  Assume min area of interest is 3.0
    number_area_bins     =   int(number_area_bins)
    range_area_bins      =   int(number_area_bins)

    freq_area_bins_pdf       =   np.zeros(number_area_bins)
    freq_area_bins_sdf       =   np.zeros(number_area_bins)
    freq_area_pdf_working    =   np.zeros(number_area_bins)
    area_array               =   np.zeros(number_area_bins)  

    for i in range(len(area_timeseries)):
    #
    #   Remember that at this point, all the below variables are string variables, not floats
    #
        area[i]         =  area_timeseries[i]
        bin_number      = int((area[i]-min_area)/bin_size + 1 )
        freq_area_bins_pdf[bin_number]    +=  1

    #.................................................................

    #   Tabulate the number of events in each bin

    for i in range(0,range_area_bins):
        for j in range(i,range_area_bins):                           # Loop over all bins to compute the GR cumulative SDF
            freq_area_bins_sdf[i] += freq_area_bins_pdf[j]
    print('')

    number_area_bins=0
    for i in range(0,range_area_bins):                               # Find the number of nonzero bins
        if freq_area_bins_sdf[i] > 0:
            number_area_bins+=1

    range_bins = int(number_area_bins)                         # Find number of nonzero bins

    log_freq_area_bins   =   np.zeros(number_area_bins)              # Define the log freq-area array

    for i in range(0,range_bins):
        if freq_area_bins_sdf[i] > 0.0:
            log_freq_area_bins[i] = -100.0                               # To get rid of it.
            log_freq_area_bins[i] = math.log10(freq_area_bins_sdf[i])     # Take the log-base-10 of the survivor function

    area_bins       =   []
    actual_area_bins=   []
    
    for i in range(0,range_bins):
        area_real = min_area + float(i)*bin_size
        area_bins.append(area_real)
        actual_area_bins.append(10**(area_real))

    #.................................................................

    if print_data:
    
        print()
        print('Actual_Area: ', actual_area_bins)
        print()
        print('Log Area', area_bins)
        print()
        print('PDF Frequency', freq_area_bins_pdf)
        print()
        print('SDF Frequency', freq_area_bins_sdf)
        print()
        print('Log Frequency Area SDF', log_freq_area_bins)
        print()

    #.................................................................

    return area_bins, log_freq_area_bins, bin_size

    ######################################################################
    
def freqMagBins(mag_timeseries, min_fit):

    #   Note that the elements in the timeseries are magnitude

    print_data = False
    
    # Create arrays of length i filled with zeros

    mag =   np.zeros(len(mag_timeseries))

    # Bins for Number-mag plot

    min_mag = 0.0
    max_mag = max(mag_timeseries)

    bin_size= 0.01

    number_mag_bins     =   (max_mag - min_mag) / bin_size + 5      #   Bin size = 0.1.  Assume min mag of interest is 3.0
    number_mag_bins     =   int(number_mag_bins)
    range_mag_bins      =   int(number_mag_bins)

    freq_mag_bins_pdf       =   np.zeros(number_mag_bins)
    freq_mag_bins_sdf       =   np.zeros(number_mag_bins)
    freq_mag_pdf_working    =   np.zeros(number_mag_bins)
    mag_array               =   np.zeros(number_mag_bins)  

    for i in range(len(mag_timeseries)):
    #
    #   Remember that at this point, all the below variables are string variables, not floats
    #
        mag[i]         =  mag_timeseries[i]
        if mag[i] >= min_mag:
            bin_number      = int((mag[i]-min_mag)/bin_size + 1 )
        try:
            freq_mag_bins_pdf[bin_number]    +=  1
        except:
            pass

    #.................................................................

    #   Tabulate the number of events in each bin

    for i in range(0,range_mag_bins):
        for j in range(i,range_mag_bins):                           # Loop over all bins to compute the GR cumulative SDF
            freq_mag_bins_sdf[i] += freq_mag_bins_pdf[j]
    print('')

    number_mag_bins=0
    for i in range(0,range_mag_bins):                               # Find the number of nonzero bins
        if freq_mag_bins_sdf[i] > 0:
            number_mag_bins+=1

    range_bins = int(number_mag_bins)                         # Find number of nonzero bins

    log_freq_mag_bins   =   np.zeros(number_mag_bins)              # Define the log freq-mag array

    for i in range(0,range_bins):
        if freq_mag_bins_sdf[i] > 0.0:
            log_freq_mag_bins[i] = -100.0                               # To get rid of it.
            log_freq_mag_bins[i] = math.log10(freq_mag_bins_sdf[i])     # Take the log-base-10 of the survivor function

    mag_bins       =   []
    actual_mag_bins=   []
    
    for i in range(0,range_bins):
        mag_real = min_mag + float(i)*bin_size
        mag_bins.append(mag_real)
        actual_mag_bins.append(10**(mag_real))

    #.................................................................

    if print_data:
    
        print()
        print('Actual_Mag: ', actual_mag_bins)
        print()
        print('Log Mag', mag_bins)
        print()
        print('PDF Frequency', freq_mag_bins_pdf)
        print()
        print('SDF Frequency', freq_mag_bins_sdf)
        print()
        print('Log Frequency Mag SDF', log_freq_mag_bins)
        print()

    #.................................................................

    return mag_bins, log_freq_mag_bins, bin_size

    ######################################################################
    
def linfit(x, y, n):

#
#     This program fits a straight line to N data.  The data are
#     assumed to be error-free.  
#
#     Definitions:
#
#     x[i]:  Abscissas of the data
#     y[i]:  Ordinates of the data
#     slope: The resulting best fit slope
#     cept:  The resulting best fit y-intercept
#     errs:  The standard error for the slope
#     errc:  The standard error for the intercept
#     n:     The exact number of data to fit
#
#
#   NOTE!!:     The *exact number* n of data to fit *must equal* = len(x) = len(y)
#               Otherwise, the code will blow up
#
#
#    n = int(len(x))

    n = int(n)

    ata = [np.zeros(2),np.zeros(2)]
    aty = [np.zeros(2),np.zeros(2)]
    atainv = [np.zeros(2),np.zeros(2)]
#    
    sumx2 = 0.
    xsum = 0.
    yxsum = 0.
    ysum = 0.

    for i in range(0,n):
        sumx2 = sumx2 + x[i] * x[i]
        xsum = xsum + x[i]
        yxsum = yxsum + y[i] * x[i]
        ysum = ysum + y[i]
#
#    ata[0][0] = sumx2
#    ata[0][1] = xsum
#    ata[1][0] = xsum
#    ata[1][1] = float(n)

    ata[0][0] = sumx2
    ata[1][0] = xsum
    ata[0][1] = xsum
    ata[1][1] = float(n)
#
    aty[0] = yxsum
    aty[1] = ysum

#
    det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    atainv[0][0] = ata[1][1]/det
    atainv[0][1] = -ata[0][1]/det
    atainv[1][0] = -ata[1][0]/det
    atainv[1][1] = ata[0][0]/det
#
    slope = atainv[0][0] * aty[0] + atainv[0][1] * aty[1]
    cept = atainv[1][0] * aty[0] + atainv[1][1] * aty[1]

    s2 = 0
    for i in range(0,n):
        s2 = s2 + (y[i] - cept - slope * x[i])**2

    s2 = s2 / (float(n) - 2.)
      
    errs = math.sqrt( float(n) * s2 / det )
    errc = math.sqrt( s2 * sumx2 / det)

#   print slope, cept, errs, errc, s2

    return (slope, cept, errs, errc, s2)

    #   Usage in calling program:  slope, cept, errs, errc, s2 = linfit(x,y)

    ######################################################################
    
def calc_neighbors(N,R,loss_fraction,KL,KC):

    number_neighbors    =  np.zeros((N,N))
    K_total             =  np.zeros((N,N))
    stress_transfer     =  np.zeros((N,N))
    
    #   Count the number of neighbors for each block [i,j]
            
    for i in range(0,N):
        for j in range(0,N):
            sum = 0.0
        
            for ki in range(i-R, i+R+1):          #   Blocks to above and below i,j
                for kj in range(j-R, j+R+1):      #   Blocks to left and right of i,j
                    if ki >= 0 and ki < N:
                        if kj >= 0 and kj < N:
                            sum += 1
                    
            number_neighbors[i,j] = sum - 1
            
    #   Calculate the total spring constant for each block
    for i in range(N):
        for j in range(N):
            if number_neighbors[i,j] > 0:
                stress_transfer[i,j] = KC*loss_fraction/number_neighbors[i,j] 
            K_total[i,j] = KL + KC*number_neighbors[i,j]
            
    return number_neighbors, stress_transfer, K_total
    
    ######################################################################
    
def smooth_stress_field(block_stress, smoothing_amplitude, N, total_failed_blocks):

    #   Adjust stress, do not increase max stress, however
    
    N2 = N*N
    
    mean_stress = np.mean(block_stress)
            
    for ki in range(N):
        for kj in range(N):
            block_stress[ki,kj] = mean_stress + (1. - smoothing_amplitude)*(block_stress[ki,kj] - mean_stress )
            
#     #   -------------------------------------------------------------

    return block_stress
    
    ######################################################################
    
def fit_line_to_mag_data(magnitude_timeseries, mag_bins, min_fit, max_fit):

    mag_bins, log_freq_mag_bins, bin_size = freqMagBins(magnitude_timeseries, min_fit)

    min_mag = min_fit  #   Since minimum area = 1, min Log_10(area) = 0.0
    max_mag = max_fit

    number_mag_bins = len(mag_bins)
#
#     Fit the line to find the b-value in the region
#

    xr    =   np.zeros(number_mag_bins)
    yr    =   np.zeros(number_mag_bins)

    #   Here us where we count the number of data to fit, then define xr[:] and yr[:]

    xr    =   np.zeros(1000)
    yr    =   np.zeros(1000)
    
    bval = 1.0
    
    print_data = False

    k = -1
    for i in range(0,number_mag_bins):
        if (mag_bins[i] >= min_mag) and (mag_bins[i] <= max_mag):
            k = k + 1
            xr[k] = mag_bins[i]
            yr[k] = log_freq_mag_bins[i]
            if print_data:
                print(' k, xr, yr: ', k, xr[k], yr[k])
# 
    kmax = k+1

    if kmax > 1:
        slope, cept, errs, errc, s2 = linfit(xr,yr, kmax)
        bval = - slope
        print()
        print('b-value for the scaling region is:')
        print(round(-bval,3),' +/- ',round(errs,3))
        print()
        print('Intercept for the scaling region is: ', round(cept,3))
        print()
        
    return slope, cept, errs, errc, s2, kmax

    ######################################################################

    
