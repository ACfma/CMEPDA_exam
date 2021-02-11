'''This program uses SVM with PCA and RFE reductions in order to analyze the \
 most important features considering an ensable of images.'''
import os
import argparse
import glob
from time import perf_counter
import logging
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from thread_pool import thread_pool
from brain_animation import brain_animation
from mean_mask import mean_mask
from roc_cv import roc_cv
#from Neuroimages_GM_AD_Detection.glass_brain import glass_brain

def vectorize_subj(images_in, mask):
    '''
    vectorize_subj return a two dimentional array of given set of images in /
    which every line correspond to an image and every row to a voxel selected/
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
        #Qui selezionare le slice di Martina
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
    names_in : list
        Array of paths to file with the same order as labels.
    '''
    zeros = np.array([1]*len(ctrl_images_in))
    ones = np.asarray([-1]*len(ad_images_in))
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
        pca = PCA()
        x_temp = pca.fit_transform(x_in)
        x_in = x_temp
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
        plt.title('AUC vs Retained Feature')
        plt.show()
        return best_n_f, best_c, fig_s
    else:
        return best_n_f, best_c, None

def rfe_pca_reductor(x_in, y_in, clf, features_r, pos_vox_r, shape_r, selector_r=None):
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
    pos_vox_r : tuple
        Coordinates of the selected voxel in images.
    shape_r : tuple
        Shape of the original image.
    selector_r : str, optional
        Selector of features choosen between 'PCA' or 'RFE'. The default is\
         None.

    Returns
    -------
    support : ndarray
        Mask of the best fitting features.
    classifier : classifier
        Selected classifier (as an object).
    zero_m : ndarray
        3D Mask of the best fitting features.
    '''
    stand_x = StandardScaler().fit_transform(x_in)
    if clf == 'SGD':
        classifier_r = SGDClassifier(class_weight='balanced', n_jobs=-1, verbose=1)

    elif clf == 'SVC':
        classifier_r = SVC(kernel='linear', class_weight='balanced', verbose=1)
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return None, None, None
    if selector_r == 'PCA':
        pca = PCA(n_components=features_r, svd_solver='randomized')#Classic RFE
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
    zero_m = np.zeros(shape_r)
    pos_1 = pos_vox_r[0][support]
    pos_2 = pos_vox_r[1][support]
    pos_3 = pos_vox_r[2][support]
    for i, _ in enumerate(pos_1):
        zero_m[pos_1[i], pos_2[i], pos_3[i]] = 1
    #Print performance time

    return support, classifier_r, zero_m
def new_data(x_in, y_in, test_set_data, test_set_lab, support, clf):
    '''
    new_data allow the reduction of the initial features to the ensamble\
     defined by the support along with the score of the fitted classifier with\
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
    clf : str
        Selected classifier between 'SVC' or 'SGD'.
    Returns
    -------
    test_x : ndarray
        Reduced test set of data.
    test_y : ndarray
        Reduced test set of labels.
    fitted_classifier : classifier
        Fitted Sklearn classifier object.
    '''
    if clf == 'SGD':
        classifier_n = SGDClassifier(class_weight='balanced', n_jobs=-1)
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

    #The same selection need to be done with  the test_X
    test_x = []
    for item in range(test_set_data.shape[0]):
        test_x.append(test_set_data[item, support])
    test_x = np.array(test_x)
    test_y = np.array(test_set_lab)
    #Resume
    print(fitted_classifier.score(test_x, test_y))
    return test_x, test_y, fitted_classifier


def spearmanr_graph(df_s, test_x, test_names_s, fitted_classifier):
    '''
    spearmanr_graph returns plotted classification of subjects along with\
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
                mmse.append(df_s['mmse'].values[i])
                if 'ad_s' in item:
                    ad_s.append(True)
                else:
                    ad_s.append(False)
    #Serve di capire come confrontare nomi nella tabella con quelli del vettore/
    #"names" splittato in modo da prendere MMSE e fare in confronto
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
    '''PATH = os.path.abspath('AD_CTRL')#Put the current path
    FILES = '*.nii'#find all nifti files with .nii in the name

    START = perf_counter()#Start the system timer

    SUBJ = glob.glob(os.path.join(PATH, FILES))

    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)

    print("Time: {}".format(perf_counter()-START))#Print performance time'''
    parser = argparse.ArgumentParser(
         description="Analyze your data using different kind of SVC with linear\
             kernel and reduction of features")
    parser.add_argument('-path', help='Path to your files', type=str)
    args = parser.parse_args()
    PATH = args.path
    FILES = r"*.nii" #find all nifti files with .nii in the name

    START = perf_counter()#Start the system timer

    SUBJ = glob.glob(os.path.join(PATH, FILES))

    AD_IMAGES, AD_NAMES, CTRL_IMAGES, CTRL_NAMES = thread_pool(SUBJ)

    print("Time: {}".format(perf_counter()-START))#Print performance time
#%%Visualize your dataset like the cool kids do, so you'll be sure of what you will be working with

    ANIM = brain_animation(sitk.GetArrayViewFromImage(CTRL_IMAGES[0]), 50, 100)
#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image

    START = perf_counter()
    IMAGES = CTRL_IMAGES.copy()
    MEAN_MASK = mean_mask(IMAGES, len(CTRL_IMAGES), overlap=0.97)
    POS_VOX = np.where(MEAN_MASK == 1)
    IMAGES.extend(AD_IMAGES.copy())
    print("Time: {}".format(perf_counter()-START))#Print performance time

#%%Select only the elements of the mask in all the images arrays
    DATASET = vectorize_subj(IMAGES, MEAN_MASK)
#%% Making labels
    LABELS, NAMES = lab_names(CTRL_IMAGES, AD_IMAGES, CTRL_NAMES, AD_NAMES)
#%% Now try a SVM-RFE
# create SVC than extract more relevant feature with selector (weigth^2)
    TRAIN_SET_DATA, TEST_SET_DATA, TRAIN_SET_LAB, TEST_SET_LAB, TRAIN_NAMES,\
    TEST_NAMES = train_test_split(DATASET, LABELS, NAMES, test_size=0.3,
                                  random_state=42)
    X, Y = TRAIN_SET_DATA, TRAIN_SET_LAB
    #%% Trying Madness for unknown reasons

    START = perf_counter()
    CLASSIFIER = SGDClassifier(class_weight='balanced', n_jobs=-1)
    #s = int(input("insert feature until stop"))
    #while (s!='stop'):
    FEATURES = [150, 100, 30]
    C = np.array([0.00001])
    STAND_X = StandardScaler().fit_transform(X)
    SELECTOR = 'PCA'
    BEST_N, CS, FIG = rfe_pca_boxplot(STAND_X, Y, 'SGD', FEATURES, C,
                                      selector_s='PCA', figure=True)
    print("Time: {}".format(perf_counter()-START))
#%% Try RFE

    #Create a matrix of zeros in witch i will change the element of the support to one
    N_FEAT = 100000
    SHAPE = (121, 145, 121)
    SUPPORT_PCA, CLASSIFIER_PCA, ZERO_M_PCA = rfe_pca_reductor(X, Y, 'SVC', 150,
                                                               POS_VOX, SHAPE,
                                                               'PCA')
    TEST_X_PCA, TEST_Y_PCA, FITTED_CLASSIFIER_PCA = new_data(X, Y, TEST_SET_DATA,
                                                             TEST_SET_LAB,
                                                             SUPPORT_PCA, 'SVC')
    FIG, AXS = plt.subplots()

    AXS.imshow(MEAN_MASK[SHAPE[0]//2, :, :], cmap='Greys_r')
    AXS.imshow(ZERO_M_PCA[SHAPE[0]//2, :, :], alpha=0.6, cmap='RdGy_r')
    #rigth now it eliminate the old X with the newer and reducted_X

    SUPPORT_RFE, CLASSIFIER_RFE, ZERO_M_RFE = rfe_pca_reductor(X, Y, 'SVC',
                                                               N_FEAT, POS_VOX,
                                                               SHAPE, 'RFE')
    TEST_X_RFE, TEST_Y_RFE, FITTED_CLASSIFIER_RFE = new_data(X, Y, TEST_SET_DATA,
                                                             TEST_SET_LAB,
                                                             SUPPORT_RFE, 'SVC')
    AXS.imshow(ZERO_M_RFE[SHAPE[0]//2, :, :], alpha=0.4, cmap='gist_gray')
    #rigth now it eliminate the old X with the newer and reducted_X

    #glass_brain(mean_mask, 0.1, 4, True, Zero_M )
#%%Distances from the Hyperplane. Due to the random nature of the import and
#split in traning set we need to search the correct elements foe spearman test
    import pandas as pd
    DF_ = pd.read_table('AD_CTRL_metadata.csv')
    FIG_PCA, RANK_PCA = spearmanr_graph(DF_, TEST_X_PCA, TEST_NAMES, FITTED_CLASSIFIER_PCA)
    FIG_RFE, RANK_RFE = spearmanr_graph(DF_, TEST_X_RFE, TEST_NAMES, FITTED_CLASSIFIER_RFE)
    plt.show()
#%%#ROC-CV
    N_SPLITS = 5
    X, Y = TEST_X_PCA, TEST_Y_PCA
    CV_ = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    CLASSIFIER = SGDClassifier(class_weight='balanced', n_jobs=-1)
    FIG, AX_ = roc_cv(X, Y, CLASSIFIER, CV_)
    X, Y = TEST_X_RFE, TEST_Y_RFE
    CV_ = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=3, random_state=42)
    CLASSIFIER = SGDClassifier(class_weight='balanced', n_jobs=-1)
    FIG, AX_ = roc_cv(X, Y, CLASSIFIER, CV_)

#%%just to see if the resize is doing well
    #glass_brain(MEAN_MASK, 0.1, 4)
