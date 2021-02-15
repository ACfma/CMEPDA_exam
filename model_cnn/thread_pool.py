'''
Module that import all nifti files from subdirectories and divide them by
name if a consistency condition is fullfilled.
'''

import logging
import os
import glob

from concurrent.futures import ThreadPoolExecutor
import SimpleITK as sitk


def thread_pool(sub):
    '''
    tread_pool creates 4 lists contanining AD and CTRL paths and images as\
     long as the single files are:
    - .nii;
    - use the above nomenclature
    - belong to a unique (or with a same name) folder.
    This function uses SimpleITK for extracting the images.

    Parameters
    ----------
    sub : Iterable of string
        Iterable containing all the paths to the nifti files.

    Returns
    -------
    ad_images : list
        List of SimpleITK.Images.
    ad_path : list
        List of the paths (string format) to the images (in the same order as\
                                                          AD_images).
    ctrl_images : list
        List of SimpleITK.Images.
    ctrl_path : list
        List of the paths (string format) to the images (in the same order as\
                                                          AD_images).
    '''
    ctrl_images = []
    ad_images = []
    ctrl_path = []
    ad_path = []

    def download(pth):
        '''
        download takes the path specified, extract the image from nifti file\
         and assign it and the path to two separate list.
        This process is used both for AD and CTRL but it could be done without\
         error only if there are no conflict in the names of folder and files\
         between paths.

        Parameters
        ----------
        x : string
            Path to the selected nifti file.

        Returns
        -------
        None.

        '''
        if pth.count('AD') > pth.count('CTRL'):
            ad_images.append(sitk.ReadImage(pth, imageIO = "NiftiImageIO"))
            ad_path.append(pth)
        elif pth.count('AD') < pth.count('CTRL'):
            ctrl_images.append(sitk.ReadImage(pth, imageIO = "NiftiImageIO"))
            ctrl_path.append(pth)
        else:
            logging.warning("An inconsistency during the import may have occured.")

    with ThreadPoolExecutor() as executor:
        executor.map(download, sub)

    return ad_images, ad_path, ctrl_images, ctrl_path

if __name__=="__main__":

    path = os.path.abspath('')#Put the current path
    FILES = r'\**\*.nii'#find all nifti files with .nii in the name
    path = path + FILES
    subj = glob.glob(os.path.normpath(path), recursive=True)

    ad, ad_names, ctrl, ctrl_names = thread_pool(subj)
