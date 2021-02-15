# -*- coding: utf-8 -*-
"""
Tools to create a confusion matrix for multilabel classification.

"""

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

def confusion_matrix(test_x, test_y, classifier):
    '''
    This function returns a confusion matrix
    plot to verify the model perfomance to predicts
    test data.

    Parameters
    ----------
    test_x : class 'numpy.ndarray'
        test data.
    test_y : class 'numpy.ndarray'
        target test data.
    classifier : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.

    Returns
    -------
    c_matrix.

    '''
    preds = classifier.predict(test_x)
    preds = np.argmax(np.round(preds),axis=1)
    score_y= np.argmax(np.round(test_y),axis=1)
    correct = np.where(preds==score_y)[0]
    print("Found %d correct labels" % len(correct))

    data = {'y_Actual': score_y,'y_Predicted': preds}
    plt.figure('model.png')
    d_frame = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    c_matrix = pd.crosstab(d_frame['y_Actual'], d_frame['y_Predicted'],\
                                   rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(c_matrix, annot=True)
    plt.show()
    return c_matrix
