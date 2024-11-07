#!/opt/local/bin python

    #   SIMSEISFileMethods.py    -   
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

def read_data_file():

    #   Read in the data. This file was written by the generate_time_series function
    #   
    
    input_file = open("../SB_Timeseries.txt", "r")
    
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
        if area > 0.001:        #   Area of an actual event has to be at least 1
            index_plate.append(int(float(items[0])))
            
            time_plate.append(float(items[1]))
            area_timeseries.append(math.log(area,10))
            
            stress_timeseries.append(float(items[3]))
            
            magnitude_timeseries.append(float(items[4]))
            
    input_file.close()
    
    
    return index_plate, time_plate, stress_timeseries, area_timeseries, magnitude_timeseries
    
    #################################################################