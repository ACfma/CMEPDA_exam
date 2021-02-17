'''
This module will create the dataset to feed the Neural Network.
'''
import os
import glob
import argparse
from time import perf_counter
import SimpleITK as sitk
from model_cnn.glass_brain import glass_brain
from model_cnn.mean_mask import mean_mask
from model_cnn.thread_pool import thread_pool
from keras.utils import to_categorical
import numpy as np

def parallelepiped(data, i, j, k, l, mask):
    '''
    Function that returns a 3D images with the selected slices i,j,k,l for each dimension.

    Parameters
    ----------
    data : list
        images.
    i : int
        First slice to be selected along x and z axis, throught this axis
        the slicing to obtain a complete brain must be coerent.
    j : int
        Last slice to be selected along x and z axis, throught this axis
        the slicing to obtain a complete brain must be coerent.
    k : First slice to be selected along y axis, is suggest to choose the
        minimal value of the mask created in model._cnn.mean_mask (approx 12).
    l : Last slice to be selected along y axis, is suggest to choose the
        maximal value of the mask created in model._cnn.mean_mask (approx 132).
    mask: int
        if mask==1 an image from array sliced is obtained, else is obtained a list
        of array.

    Returns
    -------
    images_slab : list
        images sliced.

    '''
    images_slab = []
    for img in data:
        selecting_slices = sitk.GetArrayViewFromImage(img)
        selecting_slices = selecting_slices[i:j,k:l,i:j]
        if mask==1:
            images_slab.append(sitk.GetImageFromArray(selecting_slices))
        else:
            images_slab.append(selecting_slices)
    return images_slab

def lab_names(path_ctrl, path_ad, img_ctrl, img_ad):
    '''
    lab_names create an array of labels (0,1) and paths for
    our machine learning procedure.

    Parameters
    ----------
    path_ctrl : list
        CTRL images name.
    path_ad : list
        AD images name.
    img_ctrl : list
        CTRL images.
    img_ad : list
        AD images.
    Returns
    -------
    labels : ndarray
        Array of labels.
    names : list
        list of paths to file with the same order as labels.

    '''
    names = np.append(np.array(path_ctrl), np.array(path_ad), axis=0)
    ones = np.array([1]*len(img_ctrl))
    zeros = np.array([0]*len(img_ad))
    labels = np.append(ones, zeros, axis = 0)
    return labels, names

    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create the dataset to feed\
                                     the network")
    parser.add_argument('-path', help='Path to your files', type=str)
    args = parser.parse_args()
    path = args.path
    files_name = r"\*.nii" #find all nifti files with .nii in the name

    start = perf_counter()#Start the system timer

    #Create your beginning dataset 
    subj = glob.glob(os.path.join(path, files_name))

    images_ad, names_ad, images_ctrl, names_ctrl = thread_pool(subj)

    print("Time: {}".format(perf_counter()-start))#Print performance time

    #Create array images selecting the same slice for AD and CTRL
    images_ad_sl = parallelepiped(images_ad, 11, 108, 12, 132)
    images_ad_sl = np.array(images_ad_sl)
    images_ctrl_sl = parallelepiped(images_ctrl, 11, 108, 12, 132)
    images_ctrl_sl = np.array(images_ctrl_sl)
    #If you want to see the 3D brain obtained from slicing put mask=1 in slicing and run above
    '''
    img_ctrl = sitk.GetImageFromArray(images_ctrl_sl)
    mean_mask_m = mean_mask(img_ad, len(images_ctrl_sl), overlap = 0.97)
    pos_vox = np.where(mean_mask_m == 1)
    print("Time: {}".format(perf_counter()-start))#Print performance time
    print(mean_mask_m.shape)
    glass_brain(mean_mask_m, 0.3, 3)
    '''
    #Create labels and names
    labels, names_ad_ctrl = lab_names(names_ctrl, names_ad, images_ctrl, images_ad)

    #Reshape image to feed the network
    images = []
    images.extend(images_ctrl_sl)
    images.extend(images_ad_sl)
    images = np.array(images, dtype='float32')
    images_r = images.reshape(333, 97, 120, 97, 1)
    
    #The dataset to feed the network involved images_r, categorical_labels, names_ad_ctrl
    