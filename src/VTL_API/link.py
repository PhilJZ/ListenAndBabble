#!/usr/bin/python

import sys # For argument passing
import ctypes

LIB_VTL = ctypes.cdll.LoadLibrary('src/VTL_API/VocalTractLabApi.so')

LIB_VTL.vtlGesToWav(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])#self.gesture_path,self.wav_path,self.area_path
