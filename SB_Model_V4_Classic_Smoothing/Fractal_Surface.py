#! /usr/bin/env python
from __future__ import division
from PIL import Image
# import Image
from scipy import *
import numpy as np

import scipy
from scipy import fftpack

import random

class FractalSurface(object):
#   Generate isotropic fractal surface image unp.sing spectral synthesis method [1, p.]
#   References:
#   1. Yuval Fisher, Michael McGuire,
#   The Science of Fractal Images, 1988

    def __init__(self,fd=2.5, N=16):
        self.N=N
        self.H=1-(fd-2);
        self.X=np.zeros((self.N,self.N),complex)
        self.A=np.zeros((self.N,self.N),complex)
        self.img=Image.Image()
        
        return

    def genSurface(self):
        #Spectral synthesis method
        
        N=self.N
        A=self.A
        powerr=-(self.H+1.0)/2.0

        for i in range(int(N/2)+1):
                for j in range(int(N/2)+1):
                    phase=2*pi*np.random.rand()

                    if i != 0 or j != 0:
                        rad=(i*i+j*j)**powerr*np.random.normal()
                    else:
                        rad=0.0

                    self.A[i,j]=complex(rad*np.cos(phase),rad*np.sin(phase))

                    if i == 0:
                        i0=0
                    else:
                        i0=N-i
                    if j == 0:
                        j0=0
                    else:
                        j0=N-j

                    self.A[i0,j0]=complex(rad*np.cos(phase),-rad*np.sin(phase))


        self.A.imag[int(N/2)][0]=0.0
        self.A.imag[0,int(N/2)]=0.0
        self.A.imag[int(N/2)][int(N/2)]=0.0

        for i in range(1,int(N/2)):
                for j in range(1,int(N/2)):
                        phase=2*pi*np.random.rand()
                        rad=(i*i+j*j)**powerr*np.random.normal()
                        self.A[i,N-j]=complex(rad*np.cos(phase),rad*np.sin(phase))
                        self.A[N-i,j]=complex(rad*np.cos(phase),-rad*np.sin(phase))

        itemp=fftpack.ifft2(self.A)
        itemp=itemp-itemp.min()
        
        complex_array = self.A
        mod_array = np.abs(complex_array)
        
        print('np.min_array, np.max_array: ', np.min(mod_array), np.max(mod_array))
        print('mod_array: ', mod_array)
        
        return complex_array, mod_array

    def genImage(self):
        #Aa=abs(Aa)
        Aa=self.X
#         im=Aa.real/Aa.real.max()*255.0
        im=Aa.real/Aa.real.max()*15.0
        
        self.img=Image.fromarray(uint8(im))
#         img2=Image.fromstring("L",(N,N),uint8(im).tostring())
        
        return

    def showImg(self):
        print(self.img)
        self.img.show()
        return

    def saveImg(self,fname="fs.png"):
        self.img.save(fname)
        return

    def getFSimg(self):
        return self.img

def main():
        fs=FractalSurface()
        fs.genSurface()
#         fs.genImage()
#         fs.saveImg()
#         fs.showImg()
        return

if __name__ == '__main__':
    main()