'''model_svm uses SVM with PCA and RFE reductions in order to analyze the \
 most important features considering an ensable of images. The resoults will be\
 saved as nifti images in a new (or already existing) folder named 'Masks'.'''
import os
import sys
import argparse
import glob
from time import perf_counter
import logging

import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd

from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold,  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, os.path.abspath(''))
from model_svm.thread_pool import thread_pool
#from model_svm.brain_animation import brain_animation
from model_svm.mean_mask import mean_mask
from model_svm.roc_cv import roc_cv
#from model_svm.glass_brain import glass_brain
from model_svm.cev import cum_explained_variance
from model_svm.n_comp import n_comp


def vectorize_subj(images_in, mask):
    '''
    vectorize_subj return a two dimentional array of given set of images in /
    which every line correspond to an image and every row to a voxel selected /
    by the mask used.

    Parameters
    ----------
    images_in: ndarray or list
        List of images to use.
    mask: ndarray
        Mask to apply.
    Returns
    -------
    Return a two-dimentional ndarray with shape (n_images, n_selected_voxel)
    '''
    vectors = []
    for item in images_in:
        vectors.append(sitk.GetArrayFromImage(item)[mask == 1].flatten())

    return np.array(vectors)

def lab_names(ctrl_images_in, ad_images_in, ctrl_names_in, ad_names_in):
    '''
    lab_names create an array of labels (-1,1) and paths for our machine learning procedure.

    Parameters
    ----------
    ctrl_images_in : list
        Images of CTRL.
    ad_images_in : list
        Images of AD.
    ctrl_names_in : list
        Paths of CTRL.
    ad_names_in : list
        Paths of AD.
    Returns
    -------
    labels_out : ndarray
        Array of labels.
    names_out : ndarray
        Array of paths to file with the same order as labels.
    '''
    lab_1 = np.ones(len(ctrl_images_in), dtype=int)
    lab_2 = -np.ones(len(ad_images_in), dtype=int)
    labels_out = np.append(lab_1, lab_2, axis=0)
    names_out = np.append(np.array(ctrl_names_in), np.array(ad_names_in), axis=0)
    return labels_out, names_out

def rfe_pca_boxplot(x_in, y_in, clf, features_s, c_in, selector_s=None,
                    n_splits_k=5, n_repeats=2, figure=True, random_state=42):
    '''
    rfe_pca_reductor is an iterator over a given dataset that\
     tests an confronts your estimator using roc_auc.
    The function support feature reduction based on RFE or PCA and make the\
     classification based on SGD or SVC both using a linear kernel.

    Parameters
    ----------
    x_in : ndarray
        Array of dimension (n_samples, n_features)
    y_in : ndarray
        Array of dimension (n_samples,)
    clf : string
        Estimator used as classificator. Type 'SVC' for the standard SVM with\
         linear kernel or type 'SGD' for implementig the
          Stochastic Gradient Descend on SVC.
    features_s : ndarray
        Array containing the number of feature to select.
    c_in : ndarray
        Array containing the C-value for SVM.
    selector_s: string, optional
        Strategy to follow in order to select the most important features.
        The function supports only RFE and PCA. The default is None.
    n_splits_k : int, optional
        Split for kfold cross validation. The default is 5.
    n_repeats : int, optional
        Number of repetition kfold cross validation. The default is 2.
    figure : bool, optional
        Whatever to print or not the boxplot of the resoults.
        The default is True.
    random_state : int, optional
        Random state to give to all the function that necessitate it,\
         implemeted for repetibility sake. Default it's 42.
    Returns
    -------
    best_n : int
        Optimal number of feature
    best_c : float
        Optimal C for the feature
    fig : matplotlib.Figure, optional
        If 'figure = True' returns the figure object of the boxplot.
    '''

    if clf == 'SGD':
        classifier_s = SGDClassifier(class_weight='balanced',
                                     random_state=random_state, n_jobs=-1)
    elif clf == 'SVC':
        classifier_s = SVC(kernel='linear', class_weight='balanced')
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return None, None, None
    if selector_s is None:
        models = [classifier_s]
    elif selector_s == 'RFE':
        step = float(input("Select step for RFE:"))
        models = [Pipeline(steps=[
            ('s', RFE(estimator=classifier_s, n_features_to_select=f, step=step)),
            ('m', classifier_s)]) for f in features_s]
    elif selector_s == 'PCA':
        f_max = (x_in.shape[0]*((n_splits_k-1)/n_splits_k)-(n_splits_k-1))
        if any(features_s>f_max):
            logging.warning('One or more of your input PC is out of range.\n\
                            Using allowed combinations...')
        features_s = features_s[features_s<f_max]
        models = [Pipeline(steps=[
            ('s', PCA(n_components=f)), ('m', classifier_s)]) for f in features_s]
    else:
        logging.error("Your selector is neither 'RFE' or 'PCA'")
    cvs = RepeatedStratifiedKFold(n_splits=n_splits_k, n_repeats=n_repeats,
                                  random_state=random_state)
    best_cs = []
    scores = []
    for model in models:
        if clf == 'SGD':
            param_grid = {
                'm__alpha': 1/(c_in*x_in.shape[0])
            }
        if clf == 'SVC':
            param_grid = {
                'm__C': c_in
            }
        search = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=-1)
        search.fit(x_in, y_in)
        param = list(param_grid.keys())[0]
        if clf == 'SGD':
            model.set_params(m__alpha=search.best_params_[param])
        else:
            model.set_params(m__C=search.best_params_[param])
        scores.append(cross_val_score(model, x_in, y_in, scoring='roc_auc',
                                      cv=cvs, n_jobs=-1))
        best_cs.append(c_in[search.best_index_])
        print('Done {}'.format(model))

    #Used median becouse less dependant from outlier
    median_s = [np.median(score) for score in scores]
    index = median_s.index(max(median_s))
    best_n_f = features_s[index]
    best_c = best_cs[index]
    if figure:
        fig_s = plt.figure()
        plt.boxplot(scores, sym="b", labels=features_s, patch_artist=True)
        plt.xlabel('Retained Features')
        plt.ylabel('AUC')
        plt.title('AUC vs Retained Features ({})'.format(selector_s))
        plt.show(block=False)
        return best_n_f, best_c, fig_s
    else:
        return best_n_f, best_c, None

def rfe_pca_reductor(x_in, y_in, clf, features_r, c_r, selector_r=None, random_state=42):
    '''
    rfe_pca_reductor will create a support matrix for reproducing best\
    features found. \n
    The images must be three dimensional.

    Parameters
    ----------
    x_in : ndarray
        Training set of data. Must be 2D array.
    y_in : ndarray
        Training set of labels. Must be 2D array.
    clf : str
        Selected classifier between 'SGD' and 'SVC'.
    features_r : int
        Number of selected features.
    cs : float
        Selected c for model.
    selector_r : str, optional
        Selector of features choosen between 'PCA' or 'RFE'. The default is\
         None.
    random_state : int
        Choose random state
    Returns
    -------
    support : ndarray
        Mask of the best fitting features.
    classifier : classifier
        Selected classifier (as an object).
    '''
    if clf == 'SGD':
        classifier_r = SGDClassifier(class_weight='balanced',
                                     alpha=1/(c_r*x_in.shape[0]),
                                     random_state=random_state, n_jobs=-1)

    elif clf == 'SVC':
        classifier_r = SVC(kernel='linear', C=c_r, class_weight='balanced',
                           probability=True)
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return None, None
    if selector_r == 'PCA':
        pca = PCA(n_components=features_r, svd_solver='randomized',
                  random_state=random_state)
        fit_p = pca.fit(x_in, y_in)
        diag = fit_p.explained_variance_ratio_
        indx = np.where(diag == np.max(diag))[0][0]
        feat = abs(fit_p.components_)[indx, :]
        start_int = perf_counter()
        n_feat_r = int(input("Insert number of retained features:"))
        logging.info('Time of interaction:{} s'.format(perf_counter() - start_int))
        sort_feat = np.sort(feat)[0:n_feat_r]
        support = np.in1d(feat, sort_feat)
    elif selector_r == 'RFE':
        start_int = perf_counter()
        step = float(input("Insert step for RFE:"))
        logging.info('Time of interaction:{} s'.format(perf_counter() - start_int))
        rfe = RFE(estimator=classifier_r, n_features_to_select=features_r, step=step)
        fit_r = rfe.fit(x_in, y_in)
        support = fit_r.support_

    else:
        logging.error("Your selector is neither 'RFE' or 'PCA'")
        return None, None
    #Print performance time

    return support, classifier_r
def new_data(x_in, y_in, test_set_data, test_set_lab, support, pos_vox_r, shape_r,
             classifier_n,random_state=42):
    '''
    new_data allow the reduction of the initial features to the ensamble \
    defined by the support along with the score of the fitted classifier with \
    the new set of features.

    Parameters
    ----------
    x_in : ndarray
        Training set of data. Must be a 2D array.
    y_in : ndarray
        Training set of labels. Must be a 2D array.
    test_set_data : ndarray
        Test set of data. Must be 2D a array.
    test_set_lab : ndarray
        Test set of labels. Must be 2D a array.
    support : ndarray
        1D array of selected features.
    pos_vox_r : tuple
        Coordinates of the selected voxel in images.
    shape_r : tuple
        Shape of the original image.
    classifier_n : classifier
        Selected classifier.
    random_state : int
        Select random state. Default it's 42.
    Returns
    -------
    test_x : ndarray
        Reduced test set of data.
    test_y : ndarray
        Reduced test set of labels.
    fitted_classifier : classifier
        Fitted Sklearn classifier object.
    zero_m : ndarray
        3D Mask of the best fitting features.
    '''
    red_x = []
    for item in range(x_in.shape[0]):
        red_x.append(x_in[item, support])
    red_x = np.array(red_x)
    #Fit the svc with the most important voxels
    fitted_classifier = classifier_n.fit(red_x, y_in)
    ranking = np.abs(fitted_classifier.coef_[0,:])
    #The same selection need to be done with  the test_X
    test_x = []
    for item in range(test_set_data.shape[0]):
        test_x.append(test_set_data[item, support])
    test_x = np.array(test_x)
    test_y = np.array(test_set_lab)
    #Resume
    
    score = fitted_classifier.score(test_x, test_y)

    y_pred = fitted_classifier.predict(test_x)
    conf = confusion_matrix(test_y, y_pred, labels=[-1,1]).ravel()
    
    print('Accuracy = {}'.format(score))
    print('tn = {}'.format(conf[0]))
    print('fp = {}'.format(conf[1]))
    print('fn = {}'.format(conf[2]))
    print('tp = {}'.format(conf[3]))
    plt.figure()
    sns.heatmap(conf.reshape(2,2), annot=True, xticklabels = ['AD', 'CTRL'],
                yticklabels = ['AD', 'CTRL'])
    plt.show(block = False)
    start_mat = perf_counter()
    zero_m = np.zeros(shape_r)
    pos_1 = pos_vox_r[0][support]
    pos_2 = pos_vox_r[1][support]
    pos_3 = pos_vox_r[2][support]
    for i, _ in enumerate(pos_1):
        zero_m[pos_1[i], pos_2[i], pos_3[i]] = ranking[i]
    logging.info("Matrix building time: {}".format(perf_counter()-start_mat))
    return test_x, test_y, fitted_classifier, zero_m


def spearmanr_graph(df_s, test_x, test_names_s, fitted_classifier):
    '''
    spearmanr_graph returns plotted classification of subjects along with \
    Spearman r-value and p-value.

    Parameters
    ----------
    df_s : pandas.Dataframe
        Dataframe of the information about subjects.
    test_x : ndarray
        Test set of data. Must be 2D array.
    test_names_s : ndarray or list
        Iterable of path name with the same order as test_x.
    fitted_classifier : classifier
        Fitted classifier which devide the subjects.
    Returns
    -------
    fig_s : matplotlib.Figure
        Figure object of classification.
    rank : ndarray
        Array of shape (2,) containing Spearman r-value and p-value.
    '''

    mmse = []
    ad_s = []#This will be a mask for the scatter plot
    for j in test_names_s:
        for i, item in enumerate(df_s['ID'].values):
            if item + '.' in j:
                mmse.append(df_s['MMSE'].values[i])
                if 'AD' in item:
                    ad_s.append(True)
                else:
                    ad_s.append(False)

    ad_s = np.array(ad_s)
    mmse = np.array(mmse)
    distances = fitted_classifier.decision_function(test_x)/np.sqrt(
                np.sum(np.square(fitted_classifier.coef_)))
    fig_s, ax_s = plt.subplots()
    ax_s.scatter(mmse[np.equal(ad_s, True)], distances[np.equal(ad_s, True)], cmap='b')
    ax_s.scatter(mmse[np.equal(ad_s, False)], distances[np.equal(ad_s, False)], cmap='o')
    plt.xlabel('MMSE')
    plt.ylabel('Distance from the hyperplane')
    plt.title('Distribution of subject')
    plt.legend(['AD', 'CTRL'], loc="upper left")
    plt.show(block=False)
    rank = spearmanr(mmse, distances)
    print(rank)
    return fig_s, rank


def roc_cv_trained(x_in, y_in, classifier, cvs):
    '''
    roc_cv_trained plots a mean roc curve with standard deviation along with mean auc\
     given a classifier and a cv-splitter using matplotlib.
    This version will assume the input classifier as already fitted. 
    
    Parameters
    ----------
    x_in : ndarray or list
        Data to be predicted (n_samples, n_features)
    y_in : ndarray or list
        Labels (n_samples)
    classifier : estimator
        Fitted classifier to use for the classification
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
    mean_fpr = np.linspace(0, 1, 100)#Needed for roc curve
    fig, axs = plt.subplots()
    #Here I calcoulate a lot of roc and append it to the list of resoults
    for _, test in cvs.split(x_in, y_in):
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
    plt.title('Cross-Validation ROC of fitted SVM (using the whole test set)')
    plt.legend(loc="lower right")
    plt.show()
    return fig, axs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    PATH = os.path.abspath('IMAGES/train_set')#Put the current path
    FILES = '*.nii'#find all nifti files with .nii in the name
    START = perf_counter()#Start the system timer
    SUBJ = glob.glob(os.path.join(PATH, FILES))
    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)
    print("Import time: {}".format(perf_counter()-START))#Print performance time
    # parser = argparse.ArgumentParser(
    #       description="Analyze your data using different kind of SVC with linear\
    #           kernel and reduction of features")
    # parser.add_argument('-trainpath', help='Path to your train files', type=str)
    # parser.add_argument('-testpath', help='Path to your test files', type=str)
    # args = parser.parse_args()
    # PATH = args.trainpath
    # FILES = r"*.nii"

    # START = perf_counter()#Start the system timer

    # SUBJ = glob.glob(os.path.join(PATH, FILES))

    # AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)

    # print("Import time: {}".format(perf_counter()-START))#Print performance time

    # ANIM = brain_animation(sitk.GetArrayViewFromImage(CTRL_IMAGES[0]), 50, 100)
    # plt.show(block=False)
#%% Try edge detection for mask
    IMAGES = CTRL_IMAGES.copy()
    IMAGES.extend(AD_IMAGES.copy())
    MEAN_MASK = mean_mask(IMAGES, len(CTRL_IMAGES), overlap=0.97)
    #glass_brain(mean_mask, 0.1, 4, True, Zero_M )
    POS_VOX = np.where(MEAN_MASK == 1)
#%%Select only the elements of the mask in all the images arrays
    start_glob = perf_counter()
    DATASET = vectorize_subj(IMAGES, MEAN_MASK)
#%% Making labels
    LABELS, NAMES = lab_names(CTRL_IMAGES, AD_IMAGES, CTRL_NAMES, AD_NAMES)
#%%
    TRAIN_SET_DATA, TRAIN_SET_LAB, TRAIN_NAMES = shuffle(DATASET, LABELS, NAMES,
                                                         random_state=42)
    X, Y = TRAIN_SET_DATA, TRAIN_SET_LAB
    #%%
    CLASS = input("Select Classifier between 'SVC' or 'SGD':")
    FEATURES_PCA = np.empty(0, dtype=int)
    FEATURES_RFE = np.empty(0, dtype=int)
    C = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    STAND_X = StandardScaler().fit_transform(X)
    start_pca_box = perf_counter()
    plt.figure()
    FIG = cum_explained_variance(STAND_X)
    #plt.show(block=False)
    PERC = [0.20, 0.40, 0.60, 0.80, 0.85, 0.90, 0.95]
    # QUEST = input("Do you want to use the number of PCAs at 60-70-80-85-90-95%?(Yes/No)")
    # if QUEST == 'Yes':
    for item in PERC:
        FEATURES_PCA = np.append(FEATURES_PCA, n_comp(STAND_X, item))
    # elif QUEST == 'No':
    #     CONT = 0
    #     NUM = input("Insert PCA feature n{} (ends with 'stop'):".format(CONT))
    #     while NUM!='stop':
    #         np.append(FEATURES_PCA, int(NUM))
    #         CONT = CONT+1
    #         NUM = input("Insert PCA components n{} (ends with 'stop'):".format(
    #                                                                CONT))
    # else:
    #     logging.warning('Your selection was invalid')

    BEST_N_PCA, CS_PCA, FIG_PCA = rfe_pca_boxplot(STAND_X, Y, CLASS,
                                                  FEATURES_PCA, C,
                                                  selector_s='PCA',
                                                  figure=True)
    print("Best number of PC: {}".format(BEST_N_PCA))
    # CONT = 0
    # NUM = input("Insert RFE retained feature n{} (ends with 'stop'):".format(
    #                                                                   CONT))

    plt.show(block=False)
    print("PCA boxplot's time: {}".format(perf_counter()-start_pca_box))
    start_rfe_box = perf_counter()
    # while(NUM!='stop'):
    #     FEATURES_RFE = np.append(FEATURES_RFE, int(NUM))
    #     CONT = CONT+1
    #     NUM = input("Insert RFE retained feature n{} (ends with 'stop'):".format(CONT))
    FEATURES_RFE = np.array([500000, 300000, 100000, 50000, 10000, 5000])
    BEST_N_RFE, CS_RFE, FIG_RFE = rfe_pca_boxplot(STAND_X, Y, CLASS,
                                                  FEATURES_RFE, C,
                                                  selector_s='RFE',
                                                  figure=True)
    plt.show(block=False)
    print("RFE boxplot's time: {}".format(perf_counter()-start_rfe_box))
    print("Best retained features from RFE: {}".format(BEST_N_RFE))
#%%
    #PATH = args.testpath
    PATH = os.path.abspath('IMAGES/test_set')
    SUBJ = glob.glob(os.path.join(PATH, FILES))

    AD_IMAGES_T, AD_NAMES_T, CTRL_IMAGES_T, CTRL_NAMES_T = thread_pool(SUBJ)
    IMAGES_T = CTRL_IMAGES_T.copy()
    IMAGES_T.extend(AD_IMAGES_T.copy())
    DATA = vectorize_subj(IMAGES_T, MEAN_MASK)
    LAB, NMS = lab_names(CTRL_IMAGES_T, AD_IMAGES_T, CTRL_NAMES_T, AD_NAMES_T)
    TEST_SET_DATA, TEST_SET_LAB, TEST_NAMES = shuffle(DATA, LAB, NMS,
                                                         random_state=42)
    start_pca_fred = perf_counter()
    STAND_X_TRAIN = StandardScaler().fit_transform(X)
    STAND_X_TEST = StandardScaler().fit_transform(TEST_SET_DATA)
    SHAPE = MEAN_MASK.shape
    CLASS = input("What classifier do you want to test on the reduced dataset: 'SVC' or 'SGD'?")
    print("Fitting PCA...")
    SUPPORT_PCA, CLASSIFIER_PCA = rfe_pca_reductor(STAND_X_TRAIN, Y, CLASS, BEST_N_PCA, CS_PCA, 'PCA')
    TEST_X_PCA, TEST_Y_PCA, FITTED_CLASSIFIER_PCA, M_PCA = new_data(STAND_X_TRAIN, Y,
                                                             STAND_X_TEST,
                                                             TEST_SET_LAB,
                                                             SUPPORT_PCA,
                                                             POS_VOX, SHAPE,
                                                             CLASSIFIER_PCA)

    plt.title('Confusion matrix obtained with PCA')
    print('Fit-reduction processing time for PCA: {}'.format(
                                                perf_counter()-start_pca_fred))
    start_rfe_fred = perf_counter()
    print("Fitting RFE...")
    SUPPORT_RFE, CLASSIFIER_RFE = rfe_pca_reductor(STAND_X_TRAIN, Y, CLASS, BEST_N_RFE, CS_RFE, 'RFE')
    TEST_X_RFE, TEST_Y_RFE, FITTED_CLASSIFIER_RFE, M_RFE = new_data(STAND_X_TRAIN, Y,
                                                             STAND_X_TEST,
                                                             TEST_SET_LAB,
                                                             SUPPORT_RFE,
                                                             POS_VOX, SHAPE,
                                                             CLASSIFIER_RFE)
    print("Fitting RFE...")
    plt.title('Confusion matrix obtained with RFE')
    print('Fit-reduction processing time for RFE: {}'.format(
                                                perf_counter()-start_rfe_fred))

    print('Total processing time: {}'.format(perf_counter()-start_glob))

    FIG, AXS = plt.subplots()
    plt.title('Best features found from PCA and RFE')
    AXS.imshow(MEAN_MASK[SHAPE[0]//2, :, :], cmap='Greys_r')
    AXS.imshow(M_PCA[SHAPE[0]//2, :, :], alpha=0.6, cmap='RdGy_r')
    AXS.imshow(M_RFE[SHAPE[0]//2, :, :], alpha=0.4, cmap='gist_gray')
    plt.show(block=False)


    mask_PCA = np.where(M_PCA>0,1,0)
    mask_RFE = np.where(M_RFE>0,1,0)
    sum_MASKS = mask_PCA + mask_RFE
    M_COM = np.where(sum_MASKS>1, M_PCA + M_RFE, 0)
    M_DICT = {"PCA_mask":M_PCA,
              "RFE_mask":M_RFE,
              "COMMON_mask":M_COM}
    try:
        os.chdir('Masks')
        os.chdir('..')
    except:
        os.mkdir('Masks')
    PATH = os.path.abspath('')
    PATH = os.path.join(PATH, 'Masks')
    for item in list(M_DICT.keys()):
        img = sitk.GetImageFromArray(M_DICT[item])
        sitk.WriteImage(img,os.path.join(PATH, '{}.nii'.format(item)))

#%%
    CSV_PATH = input("Insert full path of your .csv with MMSE classification of\
                     your subjects: ")
    DFM = pd.read_table(CSV_PATH)
    FIG_PCA, RANK_PCA = spearmanr_graph(DFM, TEST_X_PCA, TEST_NAMES, FITTED_CLASSIFIER_PCA)
    FIG_RFE, RANK_RFE = spearmanr_graph(DFM, TEST_X_RFE, TEST_NAMES, FITTED_CLASSIFIER_RFE)

    N_SPLITS = 5
    CVS = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    FIG, AXS = roc_cv_trained(TEST_X_PCA, TEST_Y_PCA, FITTED_CLASSIFIER_PCA, CVS)
    FIG, AXS = roc_cv(TEST_X_PCA, TEST_Y_PCA, CLASSIFIER_PCA, CVS)
    CVS = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    FIG, AXS = roc_cv_trained(TEST_X_RFE, TEST_Y_RFE, FITTED_CLASSIFIER_RFE, CVS)
    FIG, AXS = roc_cv(TEST_X_RFE, TEST_Y_RFE, CLASSIFIER_RFE, CVS)
