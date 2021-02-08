# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:54:25 2021

@author: Andrea
"""
from matplotlib import pyplot as plt


def Brain_Sequence(type_of_scan, data):
    imgs=[]
    if type_of_scan == 'Axial':
        for i, v in enumerate(data[:,0,0]):
            im = plt.imshow(data[i,:,:], animated = True)
            imgs.append([im])    
    elif type_of_scan == 'Coronal':
        for i, v in enumerate(data[0,:,0]):
            im = plt.imshow(data[:,i,:], animated = True)
            imgs.append([im])
    elif type_of_scan == 'Sagittal':
        for i, v in enumerate(data[0,0,:]):
            im = plt.imshow(data[:,:,i], animated = True)
            imgs.append([im])
    return imgs