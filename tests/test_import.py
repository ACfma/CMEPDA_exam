import unittest
import os
#import glob

import SimpleITK as sitk

from Neuroimages_GM_AD_Detection.Brain_Sequence import Brain_Sequence

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

    def test_brain_sequence(self):
        '''Test for user-defined function'''
        self.assertEqual(len(Brain_Sequence('Axial', sitk.GetArrayFromImage(self.image)[:,:,:])),121)
        self.assertEqual(len(Brain_Sequence('Coronal', sitk.GetArrayFromImage(self.image)[:,:,:])),145)
        self.assertEqual(len(Brain_Sequence('Sagittal', sitk.GetArrayFromImage(self.image)[:,:,:])),121)

if __name__=='__main__':
    unittest.main()
