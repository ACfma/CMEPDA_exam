'''
Split a certain number of images in train and test groups and save it as well as \
nifti files. The split will be executed using train_test_split function from \
scikit-learn.\n The original header is lost within the process.\n\
The files will be saved in two folder 'train_set' and 'test_set' inside the same \
along side the original images. If there are already folders with these names \
the program images will be saved in those folders.\n WARNING: When launched, \
split_folder.py will removed previous files (with the same name as the newer) \
inside "train_set" and/or "test_set" folder, if found.\n Check them before run.
'''
import os
import glob
import argparse
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from thread_pool import thread_pool
import shutil
from time import perf_counter
def split_folder(image_path):
    '''
    split_folder, given a tuple of (image, path), will save as a nifti image the\
    array extracted from the original image format.\n
    WARNING: The original file must be named as 'AD-*' or 'CTRL-*' and there must\
     be no conflict with the folder in the path.

    Parameters
    ----------
    image_path : tuple
        Tuple of (SimpleITK.IMAGE, string) where the first is the image object\
         and the second is the path to the image file.

    Returns
    -------
    None.

    '''
    arr = sitk.GetArrayFromImage(image_path[0])
    img = sitk.GetImageFromArray(arr)
    if 'AD-' in image_path[1]:
        N_AD = (image_path[1]).split('AD-')
        sitk.WriteImage(img, "AD-{}".format(N_AD[1]))
    if 'CTRL-' in image_path[1]:
        N_CTRL = (image_path[1]).split('CTRL-')
        sitk.WriteImage(img, "CTRL-{}".format(N_CTRL[1]))
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(
          description="Analyze your data using different kind of SVC with linear\
              kernel and reduction of features")
    parser.add_argument('-path', help='Path to your files', type=str)
    args = parser.parse_args()
    PATH = args.path
    FILES = r"*.nii" 

    START = perf_counter()
    
    SUBJ = glob.glob(os.path.join(PATH, FILES))

    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)
    DATASET = CTRL_IMAGES.copy()
    DATASET.extend(AD_IMAGES)
    NAMES = CTRL_NAMES.copy()
    NAMES.extend(AD_NAMES)
    TRAIN_SET_DATA, TEST_SET_DATA, TRAIN_NAMES, TEST_NAMES = train_test_split(
                                DATASET, NAMES, test_size=0.3, random_state=42)
    print("Time: {}".format(perf_counter()-START))#Print performance time
    
    TRAIN_FOLDER_NAME = 'train_set'
    TEST_FOLDER_NAME = 'test_set'
    if os.path.isdir(os.path.join(PATH, TRAIN_FOLDER_NAME)):
        rem = input("{} already exist, do you want to remove it?(Yes/No)".format(
                                                                TRAIN_FOLDER_NAME))
        if rem == 'Yes':
            pth = os.path.join(PATH, TRAIN_FOLDER_NAME)
            shutil.rmtree(pth)
            os.mkdir(os.path.join(PATH, TRAIN_FOLDER_NAME))
        else:
            print("I'm saving the files inside {}.\nAny file with the same name\
                  will be replaced...".format(TRAIN_FOLDER_NAME))
    else:
        os.mkdir(os.path.join(PATH, TRAIN_FOLDER_NAME))
    os.chdir(os.path.join(PATH, TRAIN_FOLDER_NAME))
    for item in list(zip(TRAIN_SET_DATA, TRAIN_NAMES)):
        split_folder(item)
    os.chdir(PATH)
    if os.path.isdir(os.path.join(PATH, TEST_FOLDER_NAME)):
        rem = input("{} already exist, do you want to remove it?(Yes/No)".format(
                                                                TEST_FOLDER_NAME))
        if rem == 'Yes':
            pth = os.path.join(PATH, TEST_FOLDER_NAME)
            shutil.rmtree(pth)
            os.mkdir(os.path.join(PATH, TEST_FOLDER_NAME))
        else:
            print("I'm saving the files inside {}.\nAny file with the same name\
                  will be replaced...".format(TEST_FOLDER_NAME))
    else:
        os.mkdir(os.path.join(PATH, TEST_FOLDER_NAME))
    os.chdir(os.path.join(PATH, TEST_FOLDER_NAME))
    for item in list(zip(TEST_SET_DATA, TEST_NAMES)):
        split_folder(item)
    os.chdir(PATH)
