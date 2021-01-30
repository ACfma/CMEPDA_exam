# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:34:16 2021

@author: Andrea
"""
import unittest
import os
from Neuroimages_GM_AD_Detection.Brain_Sequence import Brain_Sequence
import SimpleITK as sitk

FILE = 'testdata.html'

class TestCore(unittest.TestCase):
    '''Testing the core'''
    image = sitk.ReadImage(FILE, imageIO = "NiftiImageIO")
    
    def test_import_data(self):
        '''Testing...'''
        self.assertTupleEqual(sitk.GetArrayFromImage(self.image).shape,(121,145,121))
    
    def test_Brain_Sequence(self):
        self.assertEqual('Axial', len(Brain_Sequence(sitk.GetArrayFromImage(self.image))),1)

if __name__=='__main__':
    unittest.main()