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
    global AD_images
    AD_images.append(sitk.ReadImage(x, imageIO = "NiftiImageIO"))
    
def download_CTRL(x):
    global CTRL_images
    CTRL_images.append(sitk.ReadImage(x, imageIO = "NiftiImageIO"))
    
def Brain_Sequence(type_of_scan,data):
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
    
    start = perf_counter()#Start the system timer
    
    AD_subj = glob.glob(os.path.normpath(AD_path), recursive=True)
    CTRL_subj = glob.glob(os.path.normpath(CTRL_path), recursive=True)
    
    threads_CTRL = [thr.Thread(target=download_CTRL,
                          args=(x,)) for x in CTRL_subj]
    threads_AD = [thr.Thread(target=download_AD,
                          args=(x,)) for x in AD_subj]
    #Maybe this fart with "ifs" can be optimized. Right now it can take in any number of images of CTRL and AD 
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

    print("Time: {}".format(perf_counter()-start))#Print performance time

      
            #%%Visualize your dataset like the cool kids do, so you'll be sure of what you will be working with
    import matplotlib.animation as animation
    # plt.show()
    
    type_of_scan = input('\nType your view animation (Axial/Coronal/Sagittal): ')
    fig = plt.figure('Brain scan')
    ani = animation.ArtistAnimation(fig, Brain_Sequence(type_of_scan, sitk.GetArrayViewFromImage(CTRL_images[0])), interval=50, blit=True, repeat_delay=100)
#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image
    import multiprocessing
    CTRL_masks = []
    AD_masks = []
    def Filter_GM(x):
        #threshold_filters= sitk.YenThresholdImageFilter() #this is a good threshold too but it's a little blurry
        threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
        threshold_filters.SetInsideValue(0)#
        threshold_filters.SetOutsideValue(1)#binomial I/O
        thresh_img = threshold_filters.Execute(x)
        mask = sitk.GetArrayFromImage(thresh_img)
        #Taking GM elements
        #filtered_img = np.where(data == 1, sitk.GetArrayViewFromImage(x), data)
        return mask
    def CTRL_filtration(x):
        global CTRL_masks
        masks_img = Filter_GM(x)
        CTRL_masks.append(masks_img)
    def AD_filtration(x):
        global AD_masks
        filtered_img = Filter_GM(x)
        AD_masks.append(filtered_img)    
    # num_cores = multiprocessing.cpu_count()
    # print('N° Cores = {}'.format(num_cores))
    # start = perf_counter()
    # pool = multiprocessing.Pool(processes = num_cores)
    
    # resoults = pool.map(CTRL_filtration, CTRL_images)
    # pool.close()
    Y_N = ''
    while Y_N != 'Yes' and Y_N != 'No':
        Y_N = input("\nDo you want to apply a mean filter? \n-Type 'Yes' if you like to 'denoise' the images \n-Type 'No' for leave the images as they are.\n")
    
    if(Y_N == "Yes"):
        for x in CTRL_images:
            CTRL_filtration(x)
        for x in AD_images:
            AD_filtration(x)
        masks = []
        masks.extend(CTRL_masks)
        masks.extend(AD_masks)
        m = np.sum(np.array(masks), axis = 0)#This is a "histogram images" of occourrences of brain segmentation
        m_up = np.where(m>0.3*len(CTRL_masks), m, 0) #Alzheimer desease is diagnosticated by a loss of GM in some areas
        mean_mask = np.where(m_up > 0, 1, 0) #creating mean mask of zeros and ones
        pos_vox = np.where(mean_mask == 1)
    if(Y_N == "No"):
        for x in CTRL_images:
            CTRL_masks.append(sitk.GetArrayFromImage(x))
        for x in AD_images:
            CTRL_masks.append(sitk.GetArrayFromImage(x))
        #This part is eliminable but i left it becouse the mask based on unfiltered images is the whole images becouse they are almost zero but never zero
        masks = []
        masks.extend(CTRL_masks)
        masks.extend(AD_masks)
        m = np.sum(np.array(masks), axis = 0)#This is a "histogram images" of occourrences of brain segmentation
        m_up = np.where(m>0.001*len(CTRL_masks), m, 0) #No filter applied but it will return a ones array becouse the images are never zero
        mean_mask = np.where(m_up > 0, 1, 0) #creating mean mask of zeros and ones
        pos_vox = np.where(mean_mask == 1)
    
    
    import plotly.graph_objs as go
    fig = go.Figure(data=[go.Scatter3d(x=pos_vox[0], y=pos_vox[1], z=pos_vox[2],mode='markers', opacity=0.2)])
    
    fig.show(renderer="browser") 
    plt.imshow(mean_mask[:,50,:])
#%%Select only the elements of the mask in all the images arrays
    CTRL_vectors = []
    AD_vectors = []
    if(Y_N == "Yes"):
        for x in CTRL_images:
            CTRL_vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
        for x in AD_images:
            AD_vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
    #Right now is witnout any kind of mask (raw data if "No" is selected)
    if(Y_N == "No"):
        for x in CTRL_images:
            CTRL_vectors.append(sitk.GetArrayFromImage(x).flatten())
        for x in AD_images:
            AD_vectors.append(sitk.GetArrayFromImage(x).flatten())
    
#%% Making labels
    dataset = []
    zeros = np.array([-1]*len(CTRL_images))
    ones = np.asarray([1]*len(AD_images))
    dataset.extend(CTRL_vectors)
    dataset.extend(AD_vectors)
    dataset = np.array(dataset)
    labels = np.append(zeros, ones, axis = 0).tolist()

#%% Now try a SVM-RFE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from numpy import interp
    from sklearn.pipeline import Pipeline
    #%%One single SVC
    train_set_data, test_set_data, train_set_lab, test_set_lab = train_test_split(dataset, labels, test_size = 0.3,random_state=42)
    start = perf_counter()
    classifier = SVC(kernel='linear', probability=True)
    classifier = classifier.fit(train_set_data, train_set_lab)
    print("Time: {}".format(perf_counter()-start))#Print performance time
    coef_vect = classifier.coef_ #vettore dei pesi
    #classifier = classifier.fit(train_set_data, train_set_lab, np.abs(coef_vect))
    #coef_vect = classifier.coef_ #vettore dei pesi
     #%%Resume of above
    probs = classifier.predict_proba(test_set_data)[:, 1]#I need only positive
    fpr, tpr, _ = roc_curve(test_set_lab, probs)
    plt.plot(fpr, tpr, marker='.', label='SVC')
    print(classifier)
    print(classifier.score(test_set_data, test_set_lab))

    #%% # create SVC than extract more relevant feature with selector (weigth^2)
        #Try RFE
    from sklearn.feature_selection import RFE

    train_set_data, test_set_data, train_set_lab, test_set_lab = train_test_split(dataset, labels, test_size = 0.3,random_state=42)
    X, y = train_set_data, train_set_lab
    
    classifier = SVC(kernel='linear', probability=True)
    start = perf_counter()
    rfe = RFE(estimator=classifier, n_features_to_select=20000, step=0.5)#Classic RFE
    #rfe = RFECV(estimator=classifier, min_features_to_select=6000, step=0.5, n_jobs=-1)#Find BEST feature with cross validation, it can be paralelized so it's a little bit faster but the cross validation takes time
    FIT = rfe.fit(X,y)
    print("Time: {}".format(perf_counter()-start))#Print performance time
    #Base performance with actual dataset
    #classifier = classifier.fit(X, y)
    #print(classifier.score(test_set_data, test_set_lab))
    #rigth now it eliminate the old X with the newer and reducted_X
    
    start = perf_counter()
    red_X = []
    for x in range(X.shape[0]):
        red_X.append(X[x,FIT.support_])
    red_X = np.array(red_X)
    #Fit the svc with the most important voxels
    classifier = classifier.fit(red_X, y)
    
    #The same selection need to be done with  the test_X
    test_X = []
    for x in range(test_set_data.shape[0]):
        test_X.append(test_set_data[x,FIT.support_])
    test_X = np.array(test_X)
    test_Y = np.array(test_set_lab)
    #Resume
    print("Time: {}".format(perf_counter()-start))
    print("Score: {}".format(classifier.score(test_X, test_Y)))

    #Create a matrix of zeros in witch i will change the element of the support to one
    Sel_feat = []
    Zero_M = np.zeros((121,145,121))
    a = pos_vox[0][FIT.support_]#From some tries the pos seems to be flatten with the same order of the used vectors
    b = pos_vox[1][FIT.support_]
    c = pos_vox[2][FIT.support_]
    for i,v in enumerate(a):
        Zero_M[a[i],b[i],c[i]]=1
    fig, ax = plt.subplots()
    arr = sitk.GetArrayViewFromImage(CTRL_images[0])
    ax.imshow(arr[:,int(np.round(arr.shape[1]/2-10)),:], cmap = 'Greys_r')
    ax.imshow(Zero_M[:,int(np.round(arr.shape[1]/2-10)),:], alpha = 0.6, cmap='RdGy_r')
#%%#ROC-CV
    '''
    Is a very similar function used during the lectures but it's slimmer
    '''
    n_splits = 5
    X, Y = test_X, test_Y 
    cv = StratifiedKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    #Here I calcoulate a lot of roc and append it to the list of resoults
    for train, test in cv.split(X, Y):
        classifier.fit(X[train], Y[train])#Take train of the inputs and fit the model
        probs = classifier.predict_proba(X[test])[:, 1]#I need only positive
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
   
    #%%#
     
    def plot_cv_roc(X, y, classifier, n_splits, scaler=None):
        if scaler:
            model = Pipeline([('scaler', scaler()),
                    ('classifier', classifier)])
        else:
            model = classifier

        try:
            y = y.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
    
        cv = StratifiedKFold(n_splits)
    
        tprs = [] #True positive rate
        aucs = [] #Area under the ROC Curve
        interp_fpr = np.linspace(0, 1, 100)
        plt.figure()
        i = 0
        for train, test in cv.split(X, y):
          probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
          fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    #      print(f"{fpr} - {tpr} - {thresholds}\n")
          interp_tpr = interp(interp_fpr, fpr, tpr)
          tprs.append(interp_tpr)
        
          roc_auc = auc(fpr, tpr)
          aucs.append(roc_auc)
          plt.plot(fpr, tpr, lw=1, alpha=0.3,
                  label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
          i += 1
        plt.legend()
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.show()
    
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
              label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(interp_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(interp_fpr, mean_tpr, color='b',
                label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
                lw=2, alpha=.8)
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
    
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate',fontsize=18)
        plt.ylabel('True Positive Rate',fontsize=18)
        plt.title('Cross-Validation ROC of SVM',fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        plt.show()
    
    classifier = SVC(kernel='linear', probability=True)
    plot_cv_roc(X,y, classifier, n_splits, scaler=None)