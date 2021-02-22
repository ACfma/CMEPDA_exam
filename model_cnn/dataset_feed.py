'''
Example of the module used to create the dataset to feed the Neural Network in
model_cnn.ipynb
'''
import os
import glob
import argparse
from time import perf_counter
import SimpleITK as sitk
import numpy as np
from model_cnn.thread_pool import thread_pool


def parallelepiped(image, prllpd_img, prllpd):
    '''
    Function that returns a 3D images of 104x123x99 with the selected slices for
    each dimension and the corrispective array. Values of slices to be selected
    are 0:104 for x axis, 11:134 for y axis, 10:109 for z axis, obtained using
    mean_mask function.

    Parameters
    ----------
    image : list
        all dataset images.
    prllpd_img : list
        images that will be converted in array.
    prllpd : list
        parallelepiped images.

    Returns
    -------
    prllpd_img : list
        images sliced in array view.
    prllpd : list
        images sliced.

    '''
    for img in image:
        selecting_slices = sitk.GetArrayViewFromImage(img)
        selecting_slices = selecting_slices[0:104,11:134,10:109]
        prllpd_img.append(selecting_slices)
        prllpd.append(sitk.GetImageFromArray(selecting_slices))
    return prllpd_img, prllpd

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
    labels : array
        Array of labels.
    names : list
        list of paths to file with the same order as labels.

    '''
    names = np.append(np.array(path_ctrl), np.array(path_ad), axis=0)
    zeros = np.array([0]*len(img_ctrl))
    ones = np.array([1]*len(img_ad))
    label = np.append(zeros, ones, axis = 0)
    return label, names

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create the dataset to feed\
                                     the network")
    parser.add_argument('-path', help='Path to your files', type=str)
    args = parser.parse_args()
    path = args.path

    start = perf_counter()#Start the system timer

    #Create your beginning dataset
    subj = glob.glob(os.path.join(path, r"\*.nii"))

    images_ad, names_ad, images_ctrl, names_ctrl = thread_pool(subj)

    print("Time: {}".format(perf_counter()-start))#Print performance time

    #Create array images selecting the same slice for AD and CTRL
    images_AD, images_ADp = [], []
    images_AD, images_ADp = parallelepiped(images_ad, images_AD, images_ADp)

    images_CTRL, images_CTRLp = [], []
    images_CTRL, images_CTRLp = parallelepiped(images_ctrl, images_CTRL, images_CTRLp)
    
    #Create labels and names
    labels, names_ad_ctrl = lab_names(names_ctrl, names_ad, images_ctrl, images_ad)

    #Reshape image to feed the network
    images = []
    images.extend(images_CTRL)
    images.extend(images_AD)
    images = np.array(images, dtype='float32')
    images_r = images.reshape(333, 97, 120, 97, 1)

    #The dataset to feed the network involved images_r, labels, names_ad_ctrl
    