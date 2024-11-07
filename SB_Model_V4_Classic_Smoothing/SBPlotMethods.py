#!/opt/local/bin python

    #   Present code origin date: June 2, 2021.  
    #
    #   SBPlotMethods.py  Plotting data obtained in the SB model.
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

    #################################################################
    ################################################################

def plot_timeseries_stress_area(index_plate, time_plate, stress_timeseries, area_timeseries, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            KL, KC, max_cycles, smoothing_amplitude):
            
    #.................................................................
    
    fig, ax1 = plt.subplots()
    
    lns1 = ax1.plot(time_plate, stress_timeseries, '-', lw=1.0, color='b', zorder=1, label='Block Stress')
    
#     plt.legend()
    
    xmin,xmax  = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    
    SupTitle_text = 'SB Model Event Stress vs. Time'
    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text =  'Range of Interaction (R) = ' + str(R) + '.   Grid Size (N) = ' + str(N) + '.   Total Events = ' + \
            str(len(time_plate)) + '.'
    plt.title(Title_text, fontsize=9)

    ax1.set_ylabel('Blue Curve: Average Stress', fontsize = 12)
    ax1.set_xlabel('Time', fontsize = 12)    
    
    #.................................................................
    
    #   Second sub-plot:  Event Area vs. Time
    
    ax2 = ax1.twinx()
    
    lns2 = ax2.plot(time_plate, area_timeseries, 'ro', ms=2, zorder=2, label='Event Area')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Red Dots: Log$_{10}$ (Event Area)', fontsize=12)
#     ax1.tick_params('y', colors='k')

    # Legend
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=2, framealpha=1.0)
#     plt.legend()

    #.................................................................
    
    textstr =   'Loading: ' + 'Plate' + \
                '\nKL: ' + str(KL) + \
                '\nKC: ' + str(KC) + \
                '\nMax Cycles: ' + str(max_cycles) + \
                '\nSmoothing: ' + str(smoothing_amplitude) + \
                '\nThreshold Type: ' + threshold_type + \
                '\nThreshold Mean: ' + str(round(threshold_mean,4)) + \
                '\nThreshold Sigma: ' + str(round(threshold_deviation,4)) + \
                '\nLoss Fraction: ' + str(round(100.0*loss_fraction,2)) + '%'\
                '\nPlate Rate: ' + str(plate_rate)

#   These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.7)

#     # place a text box in lower right in axes coords
    ax2.text(0.025, 0.025, textstr, transform=ax2.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8, zorder = 3)
        
    #.................................................................
    
    figure_name = 'Stress-Time_Area_R' + str(R) + '_DF' + str(DF) + '_N' + str(N) + '_' + 'KL' + str(KL) + \
            '_KC' + str(KC) + '_' + threshold_type + '.png'
            
    plt.savefig(figure_name, dpi=300)
    

    return
    
    #################################################################
    
def plot_freqArea(area_timeseries, area_bins, log_freq_area_bins, N, R, DF, threshold_type, threshold_mean, threshold_deviation, \
        loss_fraction, plate_rate, min_fit, max_fit, KL, KC, max_cycles, smoothing_amplitude):

    print_data = False
    
    area_bins, log_freq_area_bins, bin_size = SBCalcMethods.freqAreaBins(area_timeseries)
    
    plotmin = 0.0
    plotmax = int(log_freq_area_bins[0] + 1.0)
    
    fig, ax1 = plt.subplots()

    ax1.plot(area_bins, log_freq_area_bins, 'ro', ms=3)
    

    min_area = min_fit  #   Since minimum area = 1, min Log_10(area) = 0.0
    max_area = max_fit

    number_area_bins = len(area_bins)

    #.................................................................

#
#     Fit the line to find the b-value in the region
#

    xr    =   np.zeros(number_area_bins)
    yr    =   np.zeros(number_area_bins)

    #   Here us where we count the number of data to fit, then define xr[:] and yr[:]

    xr    =   np.zeros(1000)
    yr    =   np.zeros(1000)

    k = -1
    for i in range(0,number_area_bins):
        if (area_bins[i] >= min_area) and (area_bins[i] <= max_area):
            k = k + 1
            xr[k] = area_bins[i]
            yr[k] = log_freq_area_bins[i]
            if print_data:
                print(' k, xr, yr: ', k, xr[k], yr[k])
# 
    kmax = k+1

    if kmax > 1:
        slope, cept, errs, errc, s2 = SBCalcMethods.linfit(xr,yr, kmax)
        tau_val = - slope
        print()
        print('tau value for the scaling region is:')
        print(round(-tau_val,3),' +/- ',round(errs,3))
        print()
        print('Intercept for the scaling region is: ', round(cept,3))
        print()


    #.................................................................

    plt.ylim(plotmin, plotmax)

    SupTitle_text = 'Frequency vs. Area Scaling for SB Slider Blocks'
    plt.suptitle(SupTitle_text, fontsize=12, y=0.96)
    
    Title_text =  'Range of Interaction (R) = ' + str(R) + '.   Grid Size (N) = ' + str(N) + '.   Total Events = ' + \
            str(len(area_timeseries)) + '.'
    plt.title(Title_text, fontsize=9)

    plt.ylabel('Log$_{10}$ ( Frequency )', fontsize=12)
    plt.xlabel('Log$_{10}$ ( Area )', fontsize=12)
    
    textstr =   'Slope ' + r'$\tau$' + ' (Exponent): -' + str(round(tau_val,3)) + \
                '\nKL: ' + str(KL) + \
                '\nKC: ' + str(KC) + \
                '\nMax Cycles: ' + str(max_cycles) + \
                '\nSmoothing: ' + str(smoothing_amplitude) + \
                '\nLoading: ' + 'Plate' + \
                '\nThreshold Type: ' + threshold_type + \
                '\nThreshold Mean: ' + str(round(threshold_mean,4)) +\
                '\nThreshold Sigma: ' + str(round(threshold_deviation,4)) +\
                '\nLoss Fraction: ' + str(round(100.0*loss_fraction,2)) + '%'\
                '\nPlate Rate: ' + str(plate_rate)


    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.7)

    # Place a text box in lower right in axes coords
    ax1.text(0.025, 0.025, textstr, transform=ax1.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8, zorder = 3)

    #   Plot the best fitting line for the linear (scaling) region
    
    xmin,xmax = plt.xlim()

    if (kmax > 1):
    
        x1 = xmin                 # Note that xr[kmax-1] is the largest magnitude that was used to fit the line
        y1 = slope*xmin + cept
        x2 = xmax
        y2 = slope*xmax + cept
        plt.plot([x1, x2], [y1, y2], 'k--', lw=1.15, zorder=4)

        x1 = min_area
        y1 = slope*min_area + cept
        x2 = xr[kmax-1]
        y2 = slope*xr[kmax-1] + cept
        plt.plot([x1, x2], [y1, y2], 'k-', lw=1.15)

#     #.................................................................
# 
    figure_name = 'FAPlot_R' + str(R) + '_DF' + str(DF) + '_N' + str(N) + '_' + 'KL' + str(KL) + \
            '_KC' + str(KC)  + '_' + threshold_type + '.png'
    
    plt.savefig(figure_name, dpi=300)
    
#     plt.show()
    plt.close('all')

    
    return
    
    #################################################################
    
def plot_timeseries_stress_mag(index_plate, time_plate, stress_timeseries, magnitude_timeseries, \
            N, R, DF, threshold_type, threshold_mean, threshold_deviation, loss_fraction, plate_rate, \
            KL, KC, max_cycles, smoothing_amplitude):
            
    #.................................................................
    
    fig, ax1 = plt.subplots()
    
    lns1 = ax1.plot(time_plate, stress_timeseries, '-', lw=1.0, color='b', zorder=1, label='Block Stress')
    
#     plt.legend()
    
    xmin,xmax  = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    
    SupTitle_text = 'SB Model Event Stress vs. Time'
    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text =  'Range of Interaction (R) = ' + str(R) + '.   Grid Size (N) = ' + str(N) + '.   Total Events = ' + \
            str(len(time_plate)) + '.'
    plt.title(Title_text, fontsize=9)

    ax1.set_ylabel('Blue Curve: Average Stress', fontsize = 12)
    ax1.set_xlabel('Time', fontsize = 12)    
    
    #.................................................................
    
    #   Second sub-plot:  Event Area vs. Time
    
    ax2 = ax1.twinx()
    
    lns2 = ax2.plot(time_plate, magnitude_timeseries, 'ro', ms=2, zorder=2, label='Magnitude')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Red Dots: Magnitude', fontsize=12)
#     ax1.tick_params('y', colors='k')

    # Legend
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=2, framealpha=1.0)
#     plt.legend()

    #.................................................................
    textstr =   'Loading: ' + 'Plate' + \
                '\nKL: ' + str(KL) + \
                '\nKC: ' + str(KC) + \
                '\nMax Cycles: ' + str(max_cycles) + \
                '\nSmoothing: ' + str(smoothing_amplitude) + \
                '\nThreshold Type: ' + threshold_type + \
                '\nThreshold Mean: ' + str(round(threshold_mean,4)) + \
                '\nThreshold Sigma: ' + str(round(threshold_deviation,4)) + \
                '\nLoss Fraction: ' + str(round(100.0*loss_fraction,2)) + '%'\
                '\nPlate Rate: ' + str(plate_rate)

#   These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.7)

#     # place a text box in lower right in axes coords
    ax2.text(0.025, 0.025, textstr, transform=ax2.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8, zorder = 3)
        
    #.................................................................
    
    figure_name = 'Stress-Time_Mag_R' + str(R) + '_DF' + str(DF) + '_N' + str(N) + '_' + 'KL' + str(KL) + \
            '_KC' + str(KC) + '_' + threshold_type + '.png'
    
    plt.savefig(figure_name, dpi=300)
    

    return
    
    #################################################################
    
def plot_freqMag(magnitude_timeseries, mag_bins, log_freq_mag_bins, N, R, DF, threshold_type, threshold_mean, threshold_deviation, \
        loss_fraction, plate_rate, min_fit, max_fit, KL, KC, max_cycles, smoothing_amplitude):

    print_data = False
    
#     mag_bins, log_freq_mag_bins, bin_size = SBCalcMethods.freqMagBins(magnitude_timeseries, min_fit)
    
    plotmin = 0.0
    plotmax = int(log_freq_mag_bins[0] + 1.0)
    
    fig, ax1 = plt.subplots()

    ax1.plot(mag_bins, log_freq_mag_bins, 'ro', ms = 3)
    

    min_mag = min_fit  #   Since minimum area = 1, min Log_10(area) = 0.0
    max_mag = max_fit

    number_mag_bins = len(mag_bins)

    #.................................................................

#
#     Fit the line to find the b-value in the region
#

    xr    =   np.zeros(number_mag_bins)
    yr    =   np.zeros(number_mag_bins)

    #   Here us where we count the number of data to fit, then define xr[:] and yr[:]

    xr    =   np.zeros(1000)
    yr    =   np.zeros(1000)
    
    bval = 1.0

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
        slope, cept, errs, errc, s2 = SBCalcMethods.linfit(xr,yr, kmax)
        bval = - slope
        print()
        print('b-value for the scaling region is:')
        print(round(-bval,3),' +/- ',round(errs,3))
        print()
        print('Intercept for the scaling region is: ', round(cept,3))
        print()


    #.................................................................

    plt.ylim(plotmin, plotmax)

    SupTitle_text = 'Frequency vs. Magnitude (GR) Scaling for SB Slider Blocks'
    plt.suptitle(SupTitle_text, fontsize=12, y=0.96)
    
    Title_text =  'Range of Interaction (R) = ' + str(R) + '.   Grid Size (N) = ' + str(N) + '.   Total Events = ' + \
            str(len(magnitude_timeseries)) + '.'
    plt.title(Title_text, fontsize=9)

    plt.ylabel('Log$_{10}$ ( Frequency )', fontsize=12)
    plt.xlabel('Magnitude', fontsize=12)
    
    textstr =   'Slope ' + 'b-value' + ' (Exponent): -' + str(round(bval,3)) + \
                '\nKL: ' + str(KL) + \
                '\nKC: ' + str(KC) + \
                '\nMax Cycles: ' + str(max_cycles) + \
                '\nSmoothing: ' + str(smoothing_amplitude) + \
                '\nLoading: ' + 'Plate' + \
                '\nThreshold Type: ' + threshold_type +\
                '\nThreshold Mean: ' + str(round(threshold_mean,4)) +\
                '\nThreshold Sigma: ' + str(round(threshold_deviation,4)) +\
                '\nLoss Fraction: ' + str(round(100.0*loss_fraction,2)) + '%'\
                '\nPlate Rate: ' + str(plate_rate)


    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.7)

    # Place a text box in lower right in axes coords
    ax1.text(0.025, 0.025, textstr, transform=ax1.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8, zorder = 3)

    #   Plot the best fitting line for the linear (scaling) region
    
    xmin,xmax = plt.xlim()

    
    if (kmax > 1):
    
        x1 = xmin                 # Note that xr[kmax-1] is the largest magnitude that was used to fit the line
        y1 = slope*xmin + cept
        x2 = xmax
        y2 = slope*xmax + cept
        plt.plot([x1, x2], [y1, y2], 'k--', lw=1.15, zorder=4)
        

        x1 = min_mag
        y1 = slope*min_mag + cept
        x2 = xr[kmax-1]
        y2 = slope*xr[kmax-1] + cept
        plt.plot([x1, x2], [y1, y2], 'k-', lw=1.15)
        

#     #.................................................................
# 
    figure_name = 'GRPlot_R' + str(R) + '_DF' + str(DF) + '_N' + str(N) + '_' + 'KL' + str(KL) + \
            '_KC' + str(KC) + '_' + threshold_type + '.png'
    
    plt.savefig(figure_name, dpi=300)
    
#     plt.show()
    plt.close('all')

    
    return
    
    #################################################################
    