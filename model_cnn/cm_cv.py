# -*- coding: utf-8 -*-
"""
Example of creating Stratified K-fold confusion matrix per CNN classification
used in model_cnn.ipynb.

"""
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

def cm_cv(x_in, y_in, classifier, cvs):
    '''
    Tool to plot a mean confusion matrix with standard deviation along
    given a classifier and a cv-splitter using matplotlib.

    Parameters
    ----------
    x_in : class 'ndarray', or list
        test set, data to be predicted.
    y_in : class 'ndarray' or list
        target set, labels.
    classifier : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.
    cvs : class 'sklearn.model_selection._split.RepeatedStratifiedKFold'
        Repeat Stratified Kfold.

    Returns
    -------
    scores : class 'ndarray'
        accuracy obtained evaluating test set.
    conf : class 'ndarray'
        confusion matrix.

    '''
    score = []
    confs = []
    for train, test in cvs.split(x_in, y_in):
        score.append(classifier.evaluate(x_in[test], y_in[test])[1])
        probs= classifier.predict(x_in[test])
        preds = np.round(probs)
        preds_y = []
        for i in enumerate(preds):
            preds_y.extend(preds[i])
        preds_y = np.array(preds_y, dtype='int')
        correct = np.where(preds_y==y_in[test])[0]
        print("Found %d correct labels" % len(correct))
        confs.append(confusion_matrix(y_in[test], preds_y, labels=[0,1]).ravel())
    scores = np.array(score)
    conf = np.array(confs)
    ratio_m = np.mean(conf, axis = 0)
    ratio_std = np.std(conf, axis = 0)
    plt.figure('CM')
    plt.title('3D CNN Confusion Matrix')
    print(u'Accuracy = {mean} \u00B1 {std}'.format(mean=np.mean(scores), std = np.std(scores)))
    print(u'TN = {TN} \u00B1 {sTN}'.format(TN=ratio_m[0], sTN=ratio_std[0]))
    print(u'FP = {FP} \u00B1 {sFP}'.format(FP=ratio_m[1], sFP=ratio_std[1]))
    print(u'FN = {FN} \u00B1 {sFN}'.format(FN=ratio_m[2], sFN=ratio_std[2]))
    print(u'TP = {TP} \u00B1 {sTP}'.format(TP=ratio_m[3], sTP=ratio_std[3]))
    sn.heatmap(ratio_m.reshape(2,2), annot=True, xticklabels =\
               ['CTRL','AD'], yticklabels = ['CTRL', 'AD'])
    plt.show()
    return scores, conf
