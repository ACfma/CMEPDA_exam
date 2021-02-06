# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:59:07 2020

@description: Autoencoder for MRI volume dataset
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

import glob


import threading as thr
from time import perf_counter
from itertools import zip_longest

def download_AD(x):
    global AD_images, AD_names
    AD_images.append(sitk.ReadImage(x, imageIO = "NiftiImageIO"))
    AD_names.append(x)
def download_CTRL(x):
    global CTRL_images, CTRL_names
    CTRL_images.append(sitk.ReadImage(x, imageIO = "NiftiImageIO"))
    CTRL_names.append(x)

    
if __name__=="__main__":
    file = os.path.abspath('')#Put the current path
    AD_files = '\**\*AD*.nii'#find all nifti files with AD in the name
    AD_path = file + AD_files
    file = os.path.abspath('')
    AD_files = '\**\*CTRL*.nii'
    CTRL_path = file + AD_files
    CTRL_images = []
    AD_images = []
    CTRL_data = []
    AD_data = []
    CTRL_names = []
    AD_names = []
    start = perf_counter()#Start the system timer
    
    AD_subj = glob.glob(os.path.normpath(AD_path), recursive=True)
    CTRL_subj = glob.glob(os.path.normpath(CTRL_path), recursive=True)
    
    threads_CTRL = [thr.Thread(target=download_CTRL,
                          args=(x,)) for x in CTRL_subj]
    threads_AD = [thr.Thread(target=download_AD,
                          args=(x,)) for x in AD_subj]
    #Maybe this part with "ifs" can be optimized. Right now it can take in any number of images of CTRL and AD 
    for thread1,thread2 in zip_longest(threads_CTRL,threads_AD, fillvalue=None):
        if thread1 != None:
            thread1.start()
        if thread2 != None:
            thread2.start()
    for thread1,thread2 in zip_longest(threads_CTRL,threads_AD, fillvalue=None):
        if thread1 != None:
            thread1.join()
        if thread2 != None:
            thread2.join()
    
    # start = perf_counter()
    # CTRL_A_par =[]
    # AD_A_par = []
    # import multiprocess as mp
    # from functools import partial
    # num_cores = mp.cpu_count()
    # pool = mp.Pool(processes = num_cores)
    # CTRL_results = pool.map_async(partial(sitk.ReadImage, imageIO = "NiftiImageIO"), CTRL_subj).get()
    # AD_results = pool.map_async(partial(sitk.ReadImage, imageIO = "NiftiImageIO"), AD_subj).get()
    
    print("Time: {}".format(perf_counter()-start))#Print performance time

      
            #%%Visualize your dataset like the cool kids do, so you'll be sure of what you will be working with
    def brain_animation(Image, interval, delay):
        '''
        brain_animation will create a simple animation of the blain along the three main axes of a given nifti image
        
        Parameters
        ----------
        Image: 3-D ndarray
            Selected nifti image.
        
        interval: int 
            Time (in ms) between frames.
        
        delay: int 
            Time of sleep (in ms) before repeating the animation.
        
        Returns
        -------
        Animation: Matplotlib.animation object 
            Return Matplotlib.animation object along with the plotted animation when assigned (as specified in Matplotlib documentation).
        ''' 
        def Brain_Sequence(type_of_scan,data):
            '''
            Brain_Sequence returns a list of frames from a 3D ndarray
            Parameters
            ----------
            type_of_scan: string
                Specified view of the array: "Axial", "Coronal" or "Sagittal".
            data:3D ndarray
                Array to show
                
            Returns
            -------
                imgs: list of AxesImage
                    List of frames to be animated.
            '''
            imgs=[]
            if type_of_scan == 'Axial':
                for i, v in enumerate(data[:,0,0]):
                    im = plt.imshow(data[i,:,:], animated = True)
                    imgs.append([im])    
            elif type_of_scan == 'Coronal':
                for i, v in enumerate(data[0,:,0]):
                    im = plt.imshow(data[:,i,:], animated = True)
                    imgs.append([im])
            elif type_of_scan == 'Sagittal':
                for i, v in enumerate(data[0,0,:]):
                    im = plt.imshow(data[:,:,i], animated = True)
                    imgs.append([im])
            return imgs
        from matplotlib.animation import ArtistAnimation
        
        type_of_scan = input('\nType your view animation (Axial/Coronal/Sagittal): ')
        fig = plt.figure('Brain scan')
        return  ArtistAnimation(fig, Brain_Sequence(type_of_scan, Image), interval=interval, blit=True, repeat_delay=delay)
        
    anim = brain_animation(sitk.GetArrayViewFromImage(CTRL_images[0]), 50, 100)
#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image
    
    def mean_mask(images, ctrl):
        '''
        mean_mask creates a mean mask based on a threshold along all the images given as input in order to retain just the most important voxels selected in "Automatic" or "Manual" mode.
        Parameters
        ----------
        images : list or nd array of SimpleITK.Image
            List of images to be confronted in order to obtain a mean mask

        Returns
        -------
        mean_mask : ndarray
            Array of the same dimension of a single input image.

        '''
        
        def Filter_GM(image):
            '''
            Filter_GM uses RenyiEntropyThresholdImageFilter in order to create an ndarray
            Parameters
            ----------
            image : SimpleITK.Image
                Image to apply the filter to.

            Returns
            -------
            None.

            '''
            nonlocal masks
            threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
            threshold_filters.SetInsideValue(0)#
            threshold_filters.SetOutsideValue(1)#binomial I/O
            thresh_img = threshold_filters.Execute(image)
            mask = sitk.GetArrayFromImage(thresh_img)
            masks.append(mask)
        
        A_M = ''
        
        while A_M != 'Automatic' and A_M != 'Manual':
            A_M = input("\nWhat kind of filter do you want to apply? \n-Type 'Automatic' if you like to use an automatic filter \n-Type 'Manual' for select your threshold .\n")
        
        masks = []
        
        if(A_M == "Automatic"):
            for x in images:
                Filter_GM(x)

        if(A_M == "Manual"):
            thr = float(input("Insert your threshold:"))
            for x in images:
                masks.append(np.where(sitk.GetArrayFromImage(x)>thr,1,0))
        
        m = np.sum(np.array(masks), axis = 0)#This is a "histogram images" of occourrences of brain segmentation
        m_up = np.where(m>0.03*ctrl, m, 0) #Alzheimer desease is diagnosticated by a loss of GM in some areas
        mean_mask = np.where(m_up > 0, 1, 0) #creating mean mask of zeros and ones
        
        return np.array(mean_mask)
    
    start = perf_counter()    
    images = CTRL_images.copy()
    mean_mask = mean_mask(images, len(CTRL_images))
    pos_vox = np.where(mean_mask == 1)
    images.extend(AD_images.copy())
    print("Time: {}".format(perf_counter()-start))#Print performance time

#%%Select only the elements of the mask in all the images arrays
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
            vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
        
        return np.array(vectors)
    dataset = vectorize_subj(images, mean_mask)   
#%% Making labels
    zeros = np.array([1]*len(CTRL_images))
    ones = np.asarray([-1]*len(AD_images))
    labels = np.append(zeros, ones, axis = 0)
    names = np.append(np.array(CTRL_names), np.array(AD_names), axis = 0)

#%% Now try a SVM-RFE
# create SVC than extract more relevant feature with selector (weigth^2)
    from sklearn.model_selection import train_test_split
    train_set_data, test_set_data, train_set_lab, test_set_lab, train_names, test_names = train_test_split(dataset, labels, names, test_size = 0.3,random_state=42)
    X, y = train_set_data, train_set_lab
    #%% Trying Madness for unknown reasons
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
    
    start = perf_counter()
    classifier = SGDClassifier(class_weight='balanced', n_jobs=-1)
    features = [300000, 200000, 100000, 50000, 20000, 10000]
    #Unholy line for creating a list of models
    def reductor_RFE_PCA(X, Y, classifier, features, selector = None, n_splits=5, n_repeats=2, figure=True, info = False):
        '''
        reductor_RFE_PCA is an iterator over a given dataset that test an confront your estimator using roc_auc.
        The function support feature reduction based on RFE or PCA.

        Parameters
        ----------
        X : ndarray
            Array of dimension (n_samples, n_features)
        Y : ndarray
            Array of dimension (n_samples,)
        classifier : estimator
            Estimator used as classificator
        features : ndarray
            Array containing the number of feature to select.
        selector : selector, optional
            Strategy to follow in order to select the most important features. The function supports only RFE and PCA. The default i None.
        n_splits : int, optional
            Split for kfold cross validation. The default is 5.
        n_repeats : int, optional
            Number of repetition kfold cross validation. The default is 2.
        figure : boolean, optional
            Whatever to print or not the boxplot of the resoults. The default is True.
        info : boolean, optional
            If True print the name and parameter of the last model tested at every iteration. The default is False.

        Returns
        -------
        int, optional Figure
            Returns the optimal number of feature for maximum roc_auc. If figure == True returns also the figure object of the boxplot.

        '''
        import logging
        if selector == None:
            models = classifier
        elif selector == 'RFE':
            step = float(input("Select step for RFE:"))
            models = [Pipeline(steps=[('s',RFE(estimator=classifier, n_features_to_select=f, step=step)),('m',classifier)]) for f in features]
        elif selector == 'PCA':
            'INSERIRE PCA'
        else:
            logging.error("Your selector is neither 'RFE' or 'PCA'")
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        scores = []
        for model in models:
            scores.append(cross_val_score(model, X, Y, scoring='roc_auc', cv=cv, n_jobs=-1))
            print('Done {}'.format(model))
            
        #Used median becouse the set are little for kfold so the distribution tails could be very large this can affect the mean if we use less elements
        median_s = [np.median(score) for score in scores]
        best_n = features[median_s.index(max(median_s))]
        if figure ==True:
            fig = plt.figure()
            plt.boxplot(scores, sym = "b", labels = features, patch_artist=True)
            plt.xlabel('Retained Feature')
            plt.ylabel('AUC')
            plt.title('AUC vs Retained Feature')
            plt.show()
            return best_n, fig
        else:
            return best_n
    X, y = train_set_data, train_set_lab
    best_n, fig = reductor_RFE_PCA(X, y, classifier, features, selector ='RFE', figure = True)
    print("Time: {}".format(perf_counter()-start))
#%% Try RFE    
    n_features = best_n
    
    classifier = SVC(kernel='linear', class_weight='balanced')

    start = perf_counter()
    rfe = RFE(estimator=classifier, n_features_to_select=n_features, step=0.3)#Classic RFE
    FIT = rfe.fit(X,y)
    print("Time: {}".format(perf_counter()-start))#Print performance time

    #rigth now it eliminate the old X with the newer and reducted_X
    start = perf_counter()
    red_X = []
    for x in range(X.shape[0]):
        red_X.append(X[x,FIT.support_])
    red_X = np.array(red_X)
    #Fit the svc with the most important voxels
    classifier = SVC(kernel='linear', class_weight='balanced', probability=True)#Needed for roc curve
    classifier = classifier.fit(red_X, y)
    
    #The same selection need to be done with  the test_X
    test_X = []
    for x in range(test_set_data.shape[0]):
        test_X.append(test_set_data[x,FIT.support_])
    test_X = np.array(test_X)
    test_Y = np.array(test_set_lab)
    #Resume
    print("Time: {}".format(perf_counter()-start))
    

    #Create a matrix of zeros in witch i will change the element of the support to one
    Sel_feat = []
    Zero_M = np.zeros((121,145,121))
    a = pos_vox[0][FIT.support_]
    b = pos_vox[1][FIT.support_]
    c = pos_vox[2][FIT.support_]
    for i,v in enumerate(a):
        Zero_M[a[i],b[i],c[i]]=1
    fig, ax = plt.subplots()
    arr = sitk.GetArrayViewFromImage(CTRL_images[0])
    ax.imshow(arr[int(np.round(arr.shape[1]/2-10)),:,:], cmap = 'Greys_r')
    ax.imshow(Zero_M[int(np.round(arr.shape[1]/2-10)),:,:], alpha = 0.6, cmap='RdGy_r')


#%%Distances from the Hyperplane. Due to the random nature of the import and split in traning set we need to search the correct elements foe spearman test
    import pandas as pd
    df = pd.read_table('AD_CTRL_metadata.csv')
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
    distances = classifier.decision_function(test_X)/np.sqrt(np.sum(np.square(classifier.coef_)))  #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC.decision_function
    plt.figure()
    plt.scatter(MMSE[AD == True], distances[AD == True])
    plt.scatter(MMSE[AD == False], distances[AD == False])
    plt.xlabel('MMSE')
    plt.ylabel('Distance from the hyperplane')
    plt.title('Distribution of subject')
    plt.legend(['AD', 'CTRL'],loc="upper left")
    from scipy.stats import spearmanr
    print(spearmanr(MMSE,distances))

#%%#ROC-CV
    '''
    Is a very similar function used during the lectures but it's slimmer
    '''
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_curve, auc

    def roc_cv(X, Y, classifier, cv):
        '''
        roc_cv plots a mean roc curve with standard deviation along with mean auc given a classifier and a cv-splitter using Matplotlib
        Parameters
        ----------
        X : ndarray or list
            Data to be predicted (n_samples, n_features)
        Y : ndarray or list
            Labels (n_samples)
        classifier : estimator
            Estimator to use for the classification
        cv : model selector
            Selector used for cv splitting

        Returns
        -------
        fig : Figure
        ax : AxesSubplot
        
        Returns None if the classifer doesn't own a function that allows the implemenation of roc_curve
        '''
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)#Needed for roc curve
        fig, ax = plt.subplots()
        #Here I calcoulate a lot of roc and append it to the list of resoults
        for train, test in cv.split(X, Y):

            classifier.fit(X[train], Y[train])#Take train of the inputs and fit the model
            try:
                probs = classifier.predict_proba(X[test])#I need only positive
            except:
                try:
                    probs = classifier.decision_function(X[test])#I need only positive
                except:
                    print("No discriminating function has been found for your model.")#I need only positive
                    return None, None
            fpr, tpr, _ = roc_curve(Y[test], probs)
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
                label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
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
    
    n_splits = 5
    X, Y = test_X, test_Y 
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    classifier = SGDClassifier(class_weight='balanced', n_jobs=-1)    
    fig, ax = roc_cv(X, Y, classifier, cv)
    #%%just to see if the resize is doing well

    def glass_brain(data, opacity, surface_count):
        '''
        glass_brain allows you to see the 3D array as a rendered volume.
        Given the actual dataset, the matrix's indeces are permutated for an optimal rappresentation.
        The image will be open with the user browser.
        Parameters
        ----------
        data : ndarray
            3D array of data to rapresent.
        opacity : float
            Sets the opacity of the surface. Opacity level over 0.25 could perform as well as expected (see Plotly documentation).
        surface_count : int
            Number of isosufaces to show.High number of surfaces could leed to a saturation of memory.

        Returns
        -------
        None.

        '''
        import plotly.graph_objs as go
        x1 = np.linspace(0, data.shape[0]-1, data.shape[0])  
        y1 = np.linspace(0, data.shape[1]-1, data.shape[1])  
        z1 = np.linspace(0, data.shape[2]-1, data.shape[2])
        X, Y, Z = np.meshgrid(x1, y1, z1)#creating grid matrix
        data_ein=np.einsum('ijk->jki', data)#here i swap the two directions "x" and "z" in order to rotate the image
        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data_ein.flatten(),
        isomin=data.min(),#min value of isosurface ATTENTION: A bigger calcoulation time could bring the rendering to a runtime  error if we use the browser option
        isomax=data.max(),
        opacity=opacity, # needs to be small to see through all surfaces
        surface_count=surface_count, # needs to be a large number for good volume rendering
        caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig.show(renderer="browser") 
        
    glass_brain(mean_mask, 0.1, 4)