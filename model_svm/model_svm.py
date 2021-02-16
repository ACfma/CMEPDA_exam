'''model_svm uses SVM with PCA and RFE reductions in order to analyze the \
 most important features considering an ensable of images. The resoults will be\
 saved as nifti images in a new folder named 'Masks'.'''
import os
import argparse
import glob
from time import perf_counter, process_time
import logging
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from model_svm.thread_pool import thread_pool
from model_svm.brain_animation import brain_animation
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
    labels_in : ndarray
        Array of labels.
    names_in : ndarray
        Array of paths to file with the same order as labels.
    '''
    zeros = np.array([1]*len(ctrl_images_in))
    ones = np.array([-1]*len(ad_images_in))
    labels_in = np.append(zeros, ones, axis=0)
    names_in = np.append(np.array(ctrl_names_in), np.array(ad_names_in), axis=0)
    return labels_in, names_in

def rfe_pca_boxplot(x_in, y_in, clf, features_s, c_in, selector_s=None,
                    n_splits_k=5, n_repeats=2, figure=True, random_state=42):
    '''
    rfe_pca_reductor is an iterator over a given dataset that\
     tests an confronts your estimator using roc_auc.
    The function support feature reduction based on RFE or PCA and make the\
     classification based on SGD using a SVC with linear kernel.

    Parameters
    ----------
    x_in : ndarray
        Array of dimension (n_samples, n_features)
    y_in : ndarray
        Array of dimension (n_samples,)
    clf : string
        Estimator used as classificator. Type 'SVC' for the standard SVM with\
         linear kernel or type 'SGD' for implementig the
          Stochastic Gradient descend on SVC.
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
    fig : matplotlib.Figure,optional
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
        #pca = PCA()
        #x_temp = pca.fit_transform(x_in)
        #x_in = x_temp
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
        plt.xlabel('Retained Feature')
        plt.ylabel('AUC')
        plt.title('AUC vs Retained Feature ({})'.format(selector_s))
        plt.show()
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
        Selected classifier between 'SDG' and 'SVC'.
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
    stand_x = StandardScaler().fit_transform(x_in)
    if clf == 'SGD':
        classifier_r = SGDClassifier(class_weight='balanced',
                                     alpha=1/(c_r*stand_x.shape[0]),
                                     random_state=random_state, n_jobs=-1)

    elif clf == 'SVC':
        classifier_r = SVC(kernel='linear', C=c_r, class_weight='balanced')
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return None, None, None
    if selector_r == 'PCA':
        pca = PCA(n_components=features_r, svd_solver='randomized',
                  random_state=random_state)
        fit_p = pca.fit(stand_x, y_in)
        diag = fit_p.explained_variance_ratio_
        indx = np.where(diag == np.max(diag))[0][0]
        feat = abs(fit_p.components_)[indx, :]
        n_feat_r = int(input("Insert number of retained features:"))
        sort_feat = np.sort(feat)[0:n_feat_r]
        support = np.in1d(feat, sort_feat)
    elif selector_r == 'RFE':
        step = float(input("Insert step for RFE:"))
        rfe = RFE(estimator=classifier_r, n_features_to_select=features_r, step=step)#Classic RFE
        fit_r = rfe.fit(x_in, y_in)
        support = fit_r.support_

    else:
        logging.error("Your selector is neither 'RFE' or 'PCA'")
        return None, None, None
    #Print performance time

    return support, classifier_r
def new_data(x_in, y_in, test_set_data, test_set_lab, support, pos_vox_r, shape_r, clf,random_state=42):
    '''
    new_data allow the reduction of the initial features to the ensamble \
    defined by the support along with the score of the fitted classifier with \
    the new set of features.

    Parameters
    ----------
    x_in : ndarray
        Training set of data. Must be 2D array.
    y_in : ndarray
        Training set of labels. Must be 2D array.
    test_set_data : ndarray
        Test set of data. Must be 2D array.
    test_set_lab : ndarray
        Test set of labels. Must be 2D array.
    support : ndarray
        1D array of selected features.
    pos_vox_r : tuple
        Coordinates of the selected voxel in images.
    shape_r : tuple
        Shape of the original image.
    clf : str
        Selected classifier between 'SVC' or 'SGD'.
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
    if clf == 'SGD':
        classifier_n = SGDClassifier(class_weight='balanced',
                                     random_state=random_state, n_jobs=-1)
    elif clf == 'SVC':
        classifier_n = SVC(kernel='linear', class_weight='balanced')
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return None, None, None
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
    scores = []
    confusion = []
    for train, test in KFold(
        n_splits=5, shuffle = True, random_state=random_state).split(test_x, test_y):
        y_pred = fitted_classifier.predict(test_x[test]) 
        conf = confusion_matrix(test_y[test], y_pred, labels=[-1,1]).ravel()
        confusion.append(conf)                                 
        scores.append(fitted_classifier.score(test_x[test], test_y[test]))

    scores = np.array(scores)
    confusion = np.array(confusion)
    m_conf = np.mean(confusion, axis = 0)
    sd_conf = np.std(confusion, axis = 0)
    print('Accuracy = {} +/- {}'.format(scores.mean(), scores.std()))
    print('tn = {} +/- {}'.format(m_conf[0], sd_conf[0]))
    print('fp = {} +/- {}'.format(m_conf[1], sd_conf[1]))
    print('fn = {} +/- {}'.format(m_conf[2], sd_conf[2]))
    print('tp = {} +/- {}'.format(m_conf[3], sd_conf[3]))
    plt.figure()
    sns.heatmap(m_conf.reshape(2,2), annot=True, xticklabels = ['AD', 'CTRL'], 
                yticklabels = ['AD', 'CTRL'])
    plt.show(block = False)
    zero_m = np.zeros(shape_r)
    pos_1 = pos_vox_r[0][support]
    pos_2 = pos_vox_r[1][support]
    pos_3 = pos_vox_r[2][support]
    for i, _ in enumerate(pos_1):
        zero_m[pos_1[i], pos_2[i], pos_3[i]] = ranking[i]
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
    plt.show()
    rank = spearmanr(mmse, distances)
    print(rank)
    return fig_s, rank
if __name__ == "__main__":
    PATH = os.path.abspath('IMAGES/train_set')#Put the current path
    FILES = '*.nii'#find all nifti files with .nii in the name
    START = perf_counter()#Start the system timer
    SUBJ = glob.glob(os.path.join(PATH, FILES))
    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)
    print("Time: {}".format(perf_counter()-START))#Print performance time
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

    # print("Time: {}".format(perf_counter()-START))#Print performance time

    ANIM = brain_animation(sitk.GetArrayViewFromImage(CTRL_IMAGES[0]), 50, 100)
    plt.show(block=False)
#%% Try edge detection for mask
    IMAGES = CTRL_IMAGES.copy()
    IMAGES.extend(AD_IMAGES.copy())
    MEAN_MASK = mean_mask(IMAGES, len(CTRL_IMAGES), overlap=0.97)
    POS_VOX = np.where(MEAN_MASK == 1)
#%%Select only the elements of the mask in all the images arrays
    start_glob = process_time()
    DATASET = vectorize_subj(IMAGES, MEAN_MASK)
#%% Making labels
    LABELS, NAMES = lab_names(CTRL_IMAGES, AD_IMAGES, CTRL_NAMES, AD_NAMES)
#%%
    TRAIN_SET_DATA, TRAIN_SET_LAB, TRAIN_NAMES = shuffle(DATASET, LABELS, NAMES,
                                                         random_state=42)
    X, Y = TRAIN_SET_DATA, TRAIN_SET_LAB
    #%%
    CLASS = input("Select Classifier between 'SVC' or 'SGD':")
    FEATURES_PCA = []
    FEATURES_RFE = []
    C = np.array([0.0001, 0.001, 0.01, 1., 10, 100])
    STAND_X = StandardScaler().fit_transform(X)
    start_pca_box = process_time()
    plt.figure()
    FIG = cum_explained_variance(X)
    plt.show(block=False)
    PERC = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    QUEST = input("Do you want to use the number of PCAs at 60-70-80-85-90-95%?(Yes/No)")
    if QUEST == 'Yes':
        for item in PERC:
            FEATURES_PCA.append(n_comp(X, item))
    elif QUEST == 'No':
        CONT = 0
        NUM = input("Insert PCA feature n{} (ends with 'stop'):".format(CONT))
        while(NUM!='stop'):
            FEATURES_PCA.append(int(NUM))
            CONT = CONT+1
            NUM = input("Insert PCA components n{} (ends with 'stop'):".format(
                                                                   CONT))
    else:
        logging.warning('Your selection was invalid')
    SELECTOR = 'PCA'

    BEST_N_PCA, CS_PCA, FIG_PCA = rfe_pca_boxplot(STAND_X, Y, CLASS,
                                                  FEATURES_PCA, C,
                                                  selector_s='PCA',
                                                  figure=True)

    CONT = 0
    NUM = input("Insert RFE retained feature n{} (ends with 'stop'):".format(
                                                                      CONT))

    plt.show(block=False)
    print("PCA boxplot's time: {}".format(process_time()-start_pca_box))
    start_rfe_box = process_time()
    # while(NUM!='stop'):
    #     FEATURES_RFE.append(int(NUM))
    #     CONT = CONT+1
    #     NUM = input("Insert RFE retained feature n{} (ends with 'stop'):".format(CONT))
    FEATURES_RFE = [500000, 300000, 100000, 50000, 10000, 5000]
    BEST_N_RFE, CS_RFE, FIG_RFE = rfe_pca_boxplot(STAND_X, Y, CLASS,
                                                  FEATURES_RFE, C,
                                                  selector_s='RFE',
                                                  figure=True)
    plt.show(block=False)
    print("RFE boxplot's time: {}".format(process_time()-start_rfe_box))
#%%
    #PATH = args.testpath
    PATH = os.path.abspath('IMAGES/test_set')
    SUBJ = glob.glob(os.path.join(PATH, FILES))

    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)
    IMAGES = CTRL_IMAGES.copy()
    IMAGES.extend(AD_IMAGES.copy())
    TEST_SET_DATA = vectorize_subj(IMAGES, MEAN_MASK)
    TEST_SET_LAB, TEST_NAMES = lab_names(CTRL_IMAGES, AD_IMAGES,
                                         CTRL_NAMES, AD_NAMES)
    start_pca_fred = process_time()
    N_COMP = BEST_N_PCA
    N_FEAT = BEST_N_RFE
    SHAPE = MEAN_MASK.shape
    print("Fitting PCA...")
    SUPPORT_PCA, CLASSIFIER_PCA = rfe_pca_reductor(X, Y, CLASS, BEST_N_PCA, CS_PCA, 'PCA')
    TEST_X_PCA, TEST_Y_PCA, FITTED_CLASSIFIER_PCA, M_PCA = new_data(X, Y,
                                                             TEST_SET_DATA,
                                                             TEST_SET_LAB,
                                                             SUPPORT_PCA,
                                                             POS_VOX, SHAPE,
                                                             CLASS)
    plt.title('Confusion matrix obtained with PCA')
    print('Fit-reduction processing time for PCA: {}'.format(
                                                process_time()-start_pca_fred))
    start_rfe_fred = process_time()
    print("Fitting RFE...")
    SUPPORT_RFE, CLASSIFIER_RFE = rfe_pca_reductor(X, Y, CLASS,BEST_N_RFE, CS_RFE, 'RFE')
    TEST_X_RFE, TEST_Y_RFE, FITTED_CLASSIFIER_RFE, M_RFE = new_data(X, Y,
                                                             TEST_SET_DATA,
                                                             TEST_SET_LAB,
                                                             SUPPORT_RFE,
                                                             POS_VOX, SHAPE,
                                                             CLASS)
    plt.title('Confusion matrix obtained with RFE')
    print('Fit-reduction processing time for RFE: {}'.format(
                                                process_time()-start_rfe_fred))

    print('Total processing time: {}'.format(process_time()-start_glob))
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
    
    #glass_brain(mean_mask, 0.1, 4, True, Zero_M )
#%%
    import pandas as pd
    CSV_PATH = input("Insert full path of your .csv with MMSE classification of\
                     your subjects: ")
    DFM = pd.read_table(CSV_PATH)
    FIG_PCA, RANK_PCA = spearmanr_graph(DFM, TEST_X_PCA, TEST_NAMES, FITTED_CLASSIFIER_PCA)
    plt.show(block=False)
    FIG_RFE, RANK_RFE = spearmanr_graph(DFM, TEST_X_RFE, TEST_NAMES, FITTED_CLASSIFIER_RFE)
    plt.show(block=False)
#%%
    N_SPLITS = 5
    X, Y = TEST_X_PCA, TEST_Y_PCA
    CVS = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    FIG, AXS = roc_cv(X, Y, CLASSIFIER_PCA, CVS)
    plt.show(block=False)
    X, Y = TEST_X_RFE, TEST_Y_RFE
    CVS = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    FIG, AXS = roc_cv(X, Y, CLASSIFIER_RFE, CVS)
    plt.show(block=False)
