# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:57:32 2021

@author: Andrea
"""
import unittest
import numpy

class TestArrays(unittest.TestCase):
    '''Test class'''
    def test_where_flat(self):
        '''Testing the order of numpy outputs between function "where" and "flatten"'''
        test = numpy.zeros((121,145,121))
        test[60:80,70:90,60:80]=1
        test[65:75,75:85,60:70]=2
        pos = numpy.where(test>0)
        arr = numpy.array([test[pos[0][i],pos[1][i],pos[2][i]] for i,v in enumerate(pos[0])])
        flat = test.flatten()
        flat = flat[flat>0]
        self.assertEqual(numpy.sum(arr-flat),0.)
    
if __name__=='__main__':
    unittest.main()