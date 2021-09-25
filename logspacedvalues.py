# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:26:26 2021

@author: Zack
"""
import numpy as np
cc=0.2


upperrange=np.round((10**(np.linspace(-2,-1,num=5))*cc+cc),3) #-2 to -1 goes 10% away to 1% away from cc
lowerrange=np.sort(np.round((-10**(np.linspace(-2,-1,num=5))*cc+cc),3)) #-2 to -1 goes 10% away to 1% away from cc

out=list(lowerrange) + [cc] + list(upperrange)
print(out)