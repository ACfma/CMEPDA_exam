import unittest
import os
#import glob

import SimpleITK as sitk
from unittest.mock import patch
import matplotlib
from model_svm.brain_animation import brain_animation

def get_input(text):
    return input(text)

class TestCore(unittest.TestCase):
    '''Testing the import function'''
    PATH = os.path.abspath('')
    FILE = '/tests/smwc1CTRL-1.nii'
    FILE = PATH+FILE
    #print(glob.glob(os.path.normpath(FILE)))
    image = sitk.ReadImage(FILE, imageIO = "NiftiImageIO")

    def test_import_data(self):
        '''Testing the extraction'''
        self.assertTupleEqual(sitk.GetArrayFromImage(self.image).shape,(121,145,121))

    @patch('builtins.input', lambda *args: 'Axial')
    def test_brain_sequence_axial(self):
        '''Test for user-defined function'''
        self.assertIsInstance(brain_animation(sitk.GetArrayFromImage(self.image)[:,:,:], 50, 100), matplotlib.animation.ArtistAnimation)
    @patch('builtins.input', lambda *args: 'Sagittal')
    def test_brain_sequence_sagittal(self):
        '''Test for user-defined function'''
        self.assertIsInstance(brain_animation(sitk.GetArrayFromImage(self.image)[:,:,:], 50, 100), matplotlib.animation.ArtistAnimation)
    @patch('builtins.input', lambda *args: 'Coronal')
    def test_brain_sequence_coronal(self):
        '''Test for user-defined function'''
        self.assertIsInstance(brain_animation(sitk.GetArrayFromImage(self.image)[:,:,:], 50, 100), matplotlib.animation.ArtistAnimation)
if __name__=='__main__':
    unittest.main()
