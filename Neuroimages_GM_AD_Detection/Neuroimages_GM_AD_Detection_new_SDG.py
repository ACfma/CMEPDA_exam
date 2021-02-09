# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:59:07 2020

@description: Autoencoder for MRI volume dataset
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import argparse
import glob


from time import perf_counter
import logging

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

from thread_pool import thread_pool
from brain_animation import brain_animation
from mean_mask import mean_mask
from roc_cv import roc_cv
from glass_brain import glass_brain

def vectorize_subj(images, mask):
    '''
    vectorize_subj return a two dimentional array of given set of images in which every line correspond to an image and every row to a voxel selected by the mask used.
    
    Parameters
    ----------
    images: ndarray or list
        List of images to use.
    mask: ndarray
        Mask to apply.
    
    Returns
    -------
    Return a two-dimentional ndarray with shape (n_images, n_selected_voxel)
    '''
    vectors = []
    for x in images:
        '''Qui selezionare le slice di Martina'''
        vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
    
    return np.array(vectors)

def lab_names(CTRL_images, AD_images):
    '''
    lab_names create an array of labels (-1,1) and paths for our machine learning procedure.

    Parameters
    ----------
    CTRL_images : list
        Images of CTRL.
    AD_images : list
        Images of AD.

    Returns
    -------
    labels : ndarray
        Array of labels.
    names : list
        Array of paths to file with the same order as labels.

    '''
    zeros = np.array([1]*len(CTRL_images))
    ones = np.asarray([-1]*len(AD_images))
    labels = np.append(zeros, ones, axis = 0)
    names = np.append(np.array(CTRL_names), np.array(AD_names), axis = 0)
    return labels, names

def rfe_pca_reductor(x_in, y_in, clf, features, c_in, selector = None,
                     n_splits=5, n_repeats=2, figure=True, random_state = 42):
    '''
    rfe_pca_reductor is an iterator over a given dataset that
     tests an confronts your estimator using roc_auc.
    The function support feature reduction based on RFE or PCA and make the
     classification based on SGD using a SVC with linear kernel.

    Parameters
    ----------
    x_in : ndarray
        Array of dimension (n_samples, n_features)
    y_in : ndarray
        Array of dimension (n_samples,)
    clf : string
        Estimator used as classificator. Type 'SVC' for the standard SVM with
         linear kernel or type 'SGD' for implementig the
          Stochastic Gradient descend on SVC.
    features : ndarray
        Array containing the number of feature to select.
    c_in : ndarray
        Array containing the C-value for SVM.
    selector : string, optional
        Strategy to follow in order to select the most important features.
        The function supports only RFE and PCA. The default is None.
    n_splits : int, optional
        Split for kfold cross validation. The default is 5.
    n_repeats : int, optional
        Number of repetition kfold cross validation. The default is 2.
    figure : bools, optional
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
        classifier = SGDClassifier(class_weight='balanced',
                                   random_state=random_state, n_jobs=-1)
    elif clf == 'SVC':
        classifier = SVC(kernel='linear', class_weight='balanced')
    else:
        logging.error("The selected classifier doesn't belong to the options.")
        return
    if selector == None:
        models = [classifier]
    elif selector == 'RFE':
        step = float(input("Select step for RFE:"))
        models = [Pipeline(steps=[
            ('s',RFE(estimator=classifier, n_features_to_select=f, step=step)),
            ('m',classifier)]) for f in features]
    elif selector == 'PCA':
        pca = PCA()
        x_temp= pca.fit_transform(x_in)
        x_in = x_temp
        models = [Pipeline(steps=[('s',PCA(n_components=f)), ('m', classifier)]) for f in features]
    else:
        logging.error("Your selector is neither 'RFE' or 'PCA'")
    cvs = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
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
        param =  list(param_grid.keys())[0]
        if clf == 'SGD':
            model.set_params(m__alpha = search.best_params_[param])
        else:
            model.set_params(m__C = search.best_params_[param])
        scores.append(cross_val_score(model, x_in, y_in, scoring='roc_auc',
                                      cv=cvs, n_jobs=-1))
        best_cs.append(c_in[search.best_index_])
        print('Done {}'.format(model))

    #Used median becouse less dependant from outlier
    median_s = [np.median(score) for score in scores]
    index = median_s.index(max(median_s))
    best_n = features[index]
    best_c = best_cs[index]
    if figure == True:
        fig = plt.figure()
        plt.boxplot(scores, sym = "b", labels = features, patch_artist=True)
        plt.xlabel('Retained Feature')
        plt.ylabel('AUC')
        plt.title('AUC vs Retained Feature')
        plt.show()
        return best_n, best_c, fig
    else:
        return best_n, best_c


if __name__=="__main__":
    path = os.path.abspath('AD_CTRL')#Put the current path
    files = '*.nii'#find all nifti files with .nii in the name

    start = perf_counter()#Start the system timer

    subj = glob.glob(os.path.join(path, files))

    AD_images, AD_names, CTRL_images, CTRL_names = thread_pool(subj)

    print("Time: {}".format(perf_counter()-start))#Print performance time
    # parser = argparse.ArgumentParser(
    #     description="Analyze your data using different kind of SVC with linear\
    #         kernel and reduction of features")
    # parser.add_argument('-path', help='Path to your files', type=str)
    # args = parser.parse_args()
    # path = args.path
    # FILES = r"*.nii" #find all nifti files with .nii in the name  

    # start = perf_counter()#Start the system timer

    # subj = glob.glob(os.path.join(path, FILES))

    # AD_images, AD_names, CTRL_images, CTRL_names = thread_pool(subj)

    print("Time: {}".format(perf_counter()-start))#Print performance time
#%%Visualize your dataset like the cool kids do, so you'll be sure of what you will be working with

    anim = brain_animation(sitk.GetArrayViewFromImage(CTRL_images[0]), 50, 100)
#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image

    start = perf_counter()    
    images = CTRL_images.copy()
    mean_mask = mean_mask(images, len(CTRL_images), overlap = 0.97)
    pos_vox = np.where(mean_mask == 1)
    images.extend(AD_images.copy())
    print("Time: {}".format(perf_counter()-start))#Print performance time

#%%Select only the elements of the mask in all the images arrays
    dataset = vectorize_subj(images, mean_mask)   
#%% Making labels
    labels, names = lab_names(CTRL_images, AD_images)
#%% Now try a SVM-RFE
# create SVC than extract more relevant feature with selector (weigth^2)
    train_set_data, test_set_data, train_set_lab, test_set_lab, train_names, test_names = train_test_split(dataset, labels, names, test_size = 0.3,random_state=42)
    X, y = train_set_data, train_set_lab
    #%% Trying Madness for unknown reasons

    start = perf_counter()
    classifier = SGDClassifier(class_weight='balanced', n_jobs=-1)
    #s = int(input("insert feature until stop"))
    #while (s!='stop'):
    features = [150,100,30]
    c = np.array([0.00001])
    stand_X = StandardScaler().fit_transform(X)
    selector = 'PCA'
    best_n, cs, fig = rfe_pca_reductor(stand_X, y, 'SGD', features, c, selector ='PCA', figure = True)
    print("Time: {}".format(perf_counter()-start))
#%% Try RFE    
    def FEATURE_from_PCA_RFE(X, Y, clf, features, selector = None, figure=True):
        '''
        
        
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        clf : TYPE
            DESCRIPTION.
        features : TYPE
            DESCRIPTION.
        selector : TYPE, optional
            DESCRIPTION. The default is None.
        figure : TYPE, optional
            DESCRIPTION. The default is True.
        
        Returns
        -------
        None.
        
        '''
        train_set_data, test_set_data, train_set_lab, test_set_lab, train_names, test_names = train_test_split(dataset, labels, names, test_size = 0.4,random_state=42)
        X, y = train_set_data, train_set_lab
        stand_X = StandardScaler().fit_transform(X)
        start = perf_counter()
        #features=[150, 200]
        if clf == 'SGD':
              classifier = SGDClassifier(class_weight='balanced',
                                         n_jobs=-1)
        elif clf == 'SVC':
              classifier = SVC(kernel='linear', class_weight='balanced')
        else:
              logging.error("The selected classifier doesn't belong to the options.")
              return
        stand_X = StandardScaler().fit_transform(X)
        if selector == 'PCA':
            pca = PCA(n_components=features, svd_solver = 'randomized')#Classic RFE
            FIT = pca.fit(stand_X,y)
            diag = FIT.explained_variance_ratio_
            indx = np.where(diag == np.max(diag))[0][0]
            feat = abs( FIT.components_ )[indx,:]
            n_feat = int(input("Insert number of retained features:"))
            sort_feat = np.sort(feat)[0:n_feat] 
            support = np.in1d(feat,sort_feat)
        elif selector == 'RFE':    
            rfe = RFE(estimator=classifier, n_features_to_select=features, step=0.3)#Classic RFE
            FIT = rfe.fit(X,y)
            support = FIT.support_
        
        else:
              logging.error("Your selector is neither 'RFE' or 'PCA'")
              return
        Zero_M = np.zeros((121,145,121))
        a = pos_vox[0][support]
        b = pos_vox[1][support]
        c = pos_vox[2][support]
        for i,v in enumerate(a):
          Zero_M[a[i],b[i],c[i]]=1
        #Print performance time
        
        return support, classifier, Zero_M
    def new_data(X, y, test_set_data, test_set_lab, support, clf):
        '''
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        test_set_data : TYPE
            DESCRIPTION.
        test_set_lab : TYPE
            DESCRIPTION.
        support : TYPE
            DESCRIPTION.
        clf : TYPE
            DESCRIPTION.

        Returns
        -------
        test_X : TYPE
            DESCRIPTION.
        test_Y : TYPE
            DESCRIPTION.
        fitted_classifier : TYPE
            DESCRIPTION.

        '''
        if clf == 'SGD':
              classifier = SGDClassifier(class_weight='balanced',
                                         n_jobs=-1)
        elif clf == 'SVC':
              classifier = SVC(kernel='linear', class_weight='balanced')
        else:
              logging.error("The selected classifier doesn't belong to the options.")
              return
        start = perf_counter()
        red_X = []
        for x in range(X.shape[0]):
            red_X.append(X[x,support])
        red_X = np.array(red_X)
        #Fit the svc with the most important voxels
        fitted_classifier = classifier.fit(red_X, y)
        
        #The same selection need to be done with  the test_X
        test_X = []
        for x in range(test_set_data.shape[0]):
            test_X.append(test_set_data[x,support])
        test_X = np.array(test_X)
        test_Y = np.array(test_set_lab)
        #Resume
        print("Time: {}".format(perf_counter()-start))
        print(fitted_classifier.score(test_X,test_Y))
        return test_X, test_Y, fitted_classifier
    #Create a matrix of zeros in witch i will change the element of the support to one
    n_feat = 100000
    support_pca, classifier_pca, Zero_M_pca = FEATURE_from_PCA_RFE(X, y, 'SVC', 150, 'PCA', figure=True )
    test_x_pca, test_y_pca, fitted_classifier_pca = new_data(X, y, test_set_data, test_set_lab, support_pca, 'SVC')
    fig, ax = plt.subplots()
    arr = sitk.GetArrayViewFromImage(CTRL_images[0])
    ax.imshow(arr[:,arr.shape[1]//2,:], cmap = 'Greys_r')
    ax.imshow(Zero_M_pca[:,arr.shape[1]//2,:], alpha = 0.6, cmap='RdGy_r')
    #rigth now it eliminate the old X with the newer and reducted_X
    
    
    support_rfe, classifier_rfe, Zero_M_rfe = FEATURE_from_PCA_RFE(X, y, 'SVC', n_feat, 'RFE', figure=True )
    test_x_rfe, test_y_rfe, fitted_classifier_rfe = new_data(X, y, test_set_data, test_set_lab, support_rfe, 'SVC')
    ax.imshow(Zero_M_rfe[:,arr.shape[1]//2,:], alpha = 0.4, cmap='gist_gray')
    #rigth now it eliminate the old X with the newer and reducted_X
    
    #glass_brain(mean_mask, 0.1, 4, True, Zero_M )
#%%Distances from the Hyperplane. Due to the random nature of the import and split in traning set we need to search the correct elements foe spearman test
    import pandas as pd
    df = pd.read_table('AD_CTRL_metadata.csv')
    def spearmanr_graph(df,test_X,test_names, fitted_classifier):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        test_X : TYPE
            DESCRIPTION.
        test_names : TYPE
            DESCRIPTION.
        fitted_classifier : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        MMSE = []
        AD = []#This will be a mask for the scatter plot
        for j in test_names:
            
            for i,v in enumerate(df['ID'].values): 
                if v + '.' in j:
                    MMSE.append(df['MMSE'].values[i])
                    if 'AD' in v:
                        AD.append(True)
                    else:
                        AD.append(False)
        #Serve di capire come confrontare nomi nella tabella con quelli del vettore "names" splittato in modo da prendere MMSE e fare in confronto
        AD = np.array(AD)
        MMSE = np.array(MMSE)
        distances = fitted_classifier.decision_function(test_X)/np.sqrt(np.sum(np.square(fitted_classifier.coef_)))  #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC.decision_function
        fig, ax = plt.subplots()
        ax.scatter(MMSE[AD == True], distances[AD == True], cmap = 'b')
        ax.scatter(MMSE[AD == False], distances[AD == False], cmap = 'o')
        plt.xlabel('MMSE')
        plt.ylabel('Distance from the hyperplane')
        plt.title('Distribution of subject')
        ax.legend(['AD', 'CTRL'],loc="upper left")
        #plt.show()
        from scipy.stats import spearmanr
        rank = spearmanr(MMSE,distances)
        print(rank)
        return fig, spearmanr(MMSE,distances) 
    fig_pca, rank_pca = spearmanr_graph(df,test_x_pca, test_names, fitted_classifier_pca)
    fig_rfe, rank_rfe = spearmanr_graph(df,test_x_rfe, test_names, fitted_classifier_rfe)
    plt.show()
#%%#ROC-CV
    n_splits = 5
    X, Y = test_x_pca, test_y_pca 
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    classifier = SGDClassifier(class_weight='balanced', n_jobs=-1)    
    fig, ax = roc_cv(X, Y, classifier, cv)
    X, Y = test_x_rfe, test_y_rfe 
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    classifier = SGDClassifier(class_weight='balanced', n_jobs=-1)  
    fig, ax = roc_cv(X, Y, classifier, cv)

#%%just to see if the resize is doing well
    glass_brain(mean_mask, 0.1, 4)