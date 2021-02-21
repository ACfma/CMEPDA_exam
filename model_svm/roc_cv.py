"""
roc_cv tests the ROC curve with std. deviation using k-fold cross validation.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_cv(x_in, y_in, classifier, cvs):
    '''
    roc_cv plots a mean roc curve with standard deviation along with mean auc\
     given a classifier and a cv-splitter using matplotlib.

    Parameters
    ----------
    x_in : array or list
        Data to be predicted (n_samples, n_features)
    y_in : array or list
        Labels (n_samples)
    classifier : estimator
        Estimator to use for the classification
    cvs : model selector
        Selector used for cv splitting

    Returns
    -------
    fig : matplotlib.Figure
        Figure object, None if the classifier doesn't fit the function
    axs : AxesSubplot
        Axis object, None if the classifier doesn't fit the function
    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, axs = plt.subplots()
    for train, test in cvs.split(x_in, y_in):

        classifier.fit(x_in[train], y_in[train])
        try:
            probs = classifier.predict_proba(x_in[test])[:,1]
        except:
            try:
                probs = classifier.decision_function(x_in[test])
            except:
                print("No discriminating function has been\
                      found for your model.")
                return None, None
        fpr, tpr, _ = roc_curve(y_in[test], probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    #Plotting the base option
    axs.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Base line', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #Calculate mean and std of aucs
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
    plt.show(block=False)
    return fig, axs
