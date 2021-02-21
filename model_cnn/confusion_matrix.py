# -*- coding: utf-8 -*-
"""
Examples of tool to create a confusion matrix for multilabel classification
used in model_cnn.ipynb

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
    preds = np.round(classifier.predict(test_x))
    preds_y = []
    for i in range (len(preds)):
        preds_y.extend(preds[i])
    preds_y=np.array(preds_y)
    correct = np.where(preds_y==test_y)[0]
    print("Found %d correct labels" % len(correct))
    fig, ax = plt.subplots()
    data = {'y_Actual': test_y,'y_Predicted': preds_y}
    d_frame = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    c_matrix = pd.crosstab(d_frame['y_Actual'], d_frame['y_Predicted'],\
                                   rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(c_matrix, annot=True)
    plt.show()
    return fig, ax
