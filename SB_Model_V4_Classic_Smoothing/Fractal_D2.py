#!/opt/local/bin python

    #   Method to compute a fractal d=2 surface
    
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
    
import numpy as np
import scipy
from scipy import *
from scipy import fftpack
import random

def genSurface(N,D):
    
        #   This is the spectral synthesis/filter method of constructing a fractal
        #   Adapted from:  http://shortrecipes.blogspot.com/2008/11/python-isotropic-fractal-surface.html
        
        H=1-(D-2)   #   Hausdorff dimension
        
        X=np.zeros((N,N),complex)
        A=np.zeros((N,N),complex)
        
        powerr=-(H+1.0)/2.0

        for i in range(int(N/2)+1):
                for j in range(int(N/2)+1):
                    phase=2*pi*np.random.rand()

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
                        phase=2*pi*np.random.rand()
                        rad=(i*i+j*j)**powerr*np.random.normal()
                        A[i,N-j]=complex(rad*np.cos(phase),rad*np.sin(phase))
                        A[N-i,j]=complex(rad*np.cos(phase),-rad*np.sin(phase))

        itemp=fftpack.ifft2(A)
        itemp=itemp-itemp.min()
        
        #   The plotted surface array in the original code is the real part of A scaled to, say, 15.0
        #   i.e., im=Aa.real/Aa.real.max()*15.0, and then im is plotted.  See the original code.
        
        complex_array = itemp
#         mod_array = np.abs(complex_array)
#       fractal_surface image is defined as: im=Aa.real/Aa.real.max()*15.0
        fractal_surface = complex_array.real/complex_array.real.max()
        
        return fractal_surface
        
if __name__ == '__main__':
        
    D = 2.5 #   Fractal dimension
    N=16    #   Resulting surface will be on an N x N lattice

    fractal_surface = 10.*genSurface(N,D)
    
    print(fractal_surface)
    
    print(fractal_surface - fractal_surface.mean())
    
    
    