# -*- coding: utf-8 -*-
"""
This program will test the ROC curve with std.deviation. 
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc

def roc_cv(x_in, y_in, classifier, cvs):
    '''
    roc_cv plots a mean roc curve with standard deviation along with mean auc given a classifier and a cv-splitter using Matplotlib
    Parameters
    ----------
    x_in : ndarray or list
        Data to be predicted (n_samples, n_features)
    y_in : ndarray or list
        Labels (n_samples)
    classifier : estimator
        Estimator to use for the classification
    cvs : model selector
        Selector used for cv splitting

    Returns
    -------
    fig : matplotlib.Figure
    ax : AxesSubplot

    Returns None if the classifer doesn't own a function that allows the implemenation of roc_curve
    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)#Needed for roc curve
    fig, ax = plt.subplots()
    #Here I calcoulate a lot of roc and append it to the list of resoults
    for train, test in cvs.split(x_in, y_in):

        classifier.fit(x_in[train], y_in[train])#Take train of the inputs and fit the model
        try:
            probs = classifier.predict_proba(x_in[test])#I need only positive
        except:
            try:
                probs = classifier.decision_function(x_in[test])#I need only positive
            except:
                print("No discriminating function has been found for your model.")#I need only positive
                return None, None
        fpr, tpr, _ = roc_curve(y_in[test], probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    #Plotting the base option
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Coin Flip', alpha=.8)
    #Calculate mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr) sklearn use this but i think it's the same as below
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, 
                                                                 std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validation ROC of SVM')
    plt.legend(loc="lower right")
    plt.show()
    return fig, ax
