# -*- coding: utf-8 -*-
"""
Creating ROC curve and ROC area per CNN classification.

"""
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def roc_auc_cv(x_in, y_in, classifier, cvs, btc_sT, callback):
    '''
    Tool to plot a mean ROC curve with standard deviation along with mean AUC
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
    cvs : TYPE
        DESCRIPTION.
    btc_sT : model selector
        Selector used for cv splitting
    callback : list
        callbacks to be called during the fit.

    Returns
    -------
    fig : matplotlib.Figure
        Figure object, None if the classifier desn't fit the function'
    axs : AxesSubplot
        Axis object, None if the classifier desn't fit the function

    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)#Needed for roc curve
    fig, axs = plt.subplots()
    #Here I calcoulate a lot of roc and append it to the list of resoults
    for train, test in cvs.split(x_in, y_in):
        classifier.fit(x_in[train], y_in[train], batch_size=btc_sT, epochs=100, callbacks=[callback])#Take train of the inputs and fit the model
        probs= classifier.predict(x_in[test])
        fpr, tpr, _ = roc_curve(y_in[test], probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    #Plotting the base option
    axs.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Coin Flip', alpha=.8)
    #Calculate mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    axs.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc,
                                                                 std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axs.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validation ROC of SVM')
    plt.legend(loc="lower right")
    plt.show()
    return fig, axs