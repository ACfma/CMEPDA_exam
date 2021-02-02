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

    Y_N = ''
    while Y_N != 'Yes' and Y_N != 'No':
        Y_N = input("\nDo you want to apply a mean filter? \n-Type 'Yes' if you like to 'denoise' the images \n-Type 'No' for leave the images as they are.\n")
    start = perf_counter()
    if(Y_N == "Yes"):
        for x in CTRL_images:
            CTRL_filtration(x)
        for x in AD_images:
            AD_filtration(x)
        masks = []
        masks.extend(CTRL_masks)
        masks.extend(AD_masks)
        m = np.sum(np.array(masks), axis = 0)#This is a "histogram images" of occourrences of brain segmentation
        m_up = np.where(m>0.03*len(CTRL_masks), m, 0) #Alzheimer desease is diagnosticated by a loss of GM in some areas
        mean_mask = np.where(m_up > 0, 1, 0) #creating mean mask of zeros and ones
        pos_vox = np.where(mean_mask == 1)
    if(Y_N == "No"):
        for x in CTRL_images:
            CTRL_masks.append(np.where(sitk.GetArrayFromImage(x)>0.03,1,0))
        for x in AD_images:
            CTRL_masks.append(np.where(sitk.GetArrayFromImage(x)>0.03,1,0))
        #This part is eliminable but i left it becouse the mask based on unfiltered images is the whole images becouse they are almost zero but never zero
        masks = []
        masks.extend(CTRL_masks)
        masks.extend(AD_masks)
        m = np.sum(np.array(masks), axis = 0)#This is a "histogram images" of occourrences of brain segmentation
        m_up = np.where(m>0.3*len(CTRL_masks), m, 0) #No filter applied but it will return a ones array becouse the images are never zero
        mean_mask = np.where(m_up > 0, 1, 0) #creating mean mask of zeros and ones
        pos_vox = np.where(mean_mask == 1)
    
    print("Time: {}".format(perf_counter()-start))#Print performance time

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
            CTRL_vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
        for x in AD_images:
            AD_vectors.append(sitk.GetArrayFromImage(x)[mean_mask == 1].flatten())
    
#%% Making labels
    dataset = []
    zeros = np.array([1]*len(CTRL_images))
    ones = np.asarray([-1]*len(AD_images))
    dataset.extend(CTRL_vectors)
    dataset.extend(AD_vectors)
    dataset = np.array(dataset)
    labels = np.append(zeros, ones, axis = 0).tolist()
    names = np.append(np.array(CTRL_names), np.array(AD_names), axis = 0).tolist()
#%%Flatten act just like where? Let's see if the feature selected with pos_vox order are the same of the vectors obtained with flatten
    Sel_feat = []
    Zero_M = np.zeros((121,145,121))
    a = pos_vox[0]#From some tries the pos seems to be flatten with the same order of the used vectors
    b = pos_vox[1]
    c = pos_vox[2]
    sub = 10#number of subject selected
    arr = sitk.GetArrayViewFromImage(CTRL_images[sub])#Let's take the first one
    for i,v in enumerate(a):
        Sel_feat.append(arr[a[i],b[i],c[i]])
    p = CTRL_vectors[sub]- np.array(Sel_feat)
    if len(p[p>0])==0:
        print("They're the same array")
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
    
    start = perf_counter()
    classifier = SVC(kernel='linear', class_weight='balanced')
    features = [300000, 200000, 100000, 50000, 20000, 10000]
    #Unholy line for creating a list of models
    models = [Pipeline(steps=[('s',RFE(estimator=classifier, n_features_to_select=f, step=0.5)),('m',classifier)]) for f in features]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    X, y = train_set_data, train_set_lab
    scores = []
    for model in models:
        scores.append(cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1))
    plt.figure()
    plt.boxplot(scores, sym = "b", labels = features, patch_artist=True)
    plt.xlabel('Retained Feature')
    plt.ylabel('AUC')
    plt.title('AUC vs Retained Feature')
    plt.show()
    #Used median becouse the set are little for kfold so the distribution tails could be very large this can affect the mean if we use less elements
    median_s = [np.median(score) for score in scores]
    best_n = features[median_s.index(max(median_s))]
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
                if 'AD' in j:
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
    plt.xlabel('Distance from the hyperplane')
    plt.ylabel('MMSE')
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
    from numpy import interp
    n_splits = 5
    X, Y = test_X, test_Y 
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=42)
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