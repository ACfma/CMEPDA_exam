# -*- coding: utf-8 -*-
"""
Creating ROC curve and ROC area per multi-label classification.

"""
from itertools import cycle
import numpy as np
from sklearn.metrics import roc_curve, auc
from numpy import interp
from matplotlib import pyplot as plt


def auc_roc(test_x, test_y, classifier, n_classes):
    '''
    This function return one ROC curve drawn per label,
    and also draw a ROC curve by considering each element of the
    label indicator matrix as a binary prediction (micro-averaging) and
    another one which gives equal weight to the classification of each label
    (macro-averaging).

    Parameters
    ----------
    X : class 'numpy.ndarray'
        test data.
    y : class 'numpy.ndarray'
        target test data.
    classifier : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.
    n_classes : int
        number of classes available.

    Returns
    -------
    A ROC_AUC plot.

    '''
    # Learn to predict each class against the other
    y_score=classifier.predict(test_x)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates to compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
              label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
