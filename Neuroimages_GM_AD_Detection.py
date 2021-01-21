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

#%%Select a zone with mouse
    import cv2
    
    def draw_rect(image):
        '''Record Left mouse button interaction in order to draw a rectangle.
        Prior to that it will resize the image for an easier selection.
        Input: 2-D array
        Output: 2-D array containing the rectangle extremes' coordinates'''
        
        zoom = int(input('Image shape is: {}. Select Zoom Coefficient (es. 1 = Normal image shape): '.format(image.shape)))
        rsz = cv2.resize(image, (zoom*image.shape[0], zoom*image.shape[1]))
        rect_pts = [] # Starting and ending points
        win_name = "Selected slice" # Window name
    
        def select_points(event, x, y, flags, param):
            
            nonlocal rect_pts
            if event == cv2.EVENT_LBUTTONDOWN and len(rect_pts) == 0:
                rect_pts = [(x, y)]
    
            if event == cv2.EVENT_LBUTTONUP and len(rect_pts) == 1:
                rect_pts.append((x, y))
    
                # draw a rectangle around the region of interest
                cv2.rectangle(rsz, rect_pts[0], rect_pts[1], (255,0,0))
                cv2.imshow(win_name, rsz)
    
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, select_points)
        print("Instruction: Click with your Left mouse button to select the starting point of the selection rectangle, hold it, and drug your mouse until the end of the zone that you want to select than release it.")
        print("-Type 'r' for refreshing the image \n-Type 'c' to confirm your selection \n-Type 'e' to exit selection")
        while True:
            # display the image and wait for a keypress
            cv2.imshow(win_name,rsz)
            key = cv2.waitKey(0)  #Wait for an interaction and assign to the key an unicode value
            
            if key == ord("r"): # Hit 'r' to replot the image
                #rsz = image.copy()
                rsz = cv2.resize(image, (zoom*image.shape[0], zoom*image.shape[1]))
                rect_pts = []
    
            elif key == ord("c"): # Hit 'c' to confirm the selection
                break
            elif key == ord("e"): # Hit 'c' to exit
                cv2.destroyWindow(win_name)
                return np.array([[0,0], [image.shape[0],image.shape[1]]])
        
        # close the open windows
        cv2.destroyWindow(win_name)
        if len(rect_pts) == 0:
            return np.array([[0,0], [image.shape[0],image.shape[1]]])
        
        return np.round(np.array(rect_pts)/zoom) 
    
    Y_N = ''
    while Y_N != 'Yes' and Y_N != 'No':
        Y_N = input("\nDo you want to select a particoluar zone of the brain? \n-Type 'Yes' to select it. \n-Type 'No' for leave the images as they are.\n")
    if Y_N == 'Yes':
        selected_slice_coronal = int(input('Select coronal image to crop in order to select the first two dimention of your ROI (Coronal slices 0-{}):'.format(CTRL_images[0].GetSize()[1])))
    
        print('-'*30)
        print('Select Coronal Area')
        print('-'*30)
        points_coronal = draw_rect(sitk.GetArrayViewFromImage(CTRL_images[0])[:,selected_slice_coronal,:])
        
        selected_slice_axial = int(input('Select axial image to crop in order to select the third dimention of your ROI (Axial slices 0-{}):'.format(CTRL_images[0].GetSize()[0])))
        print('-'*30)
        print('Select Axial Area')
        print('-'*30)
        points_axial = draw_rect(sitk.GetArrayViewFromImage(CTRL_images[0])[selected_slice_axial,:,:])
        #Including the case in wich the user select the box in any possible direcion 
        if (points_coronal[0,1]!= None or points_coronal[1,1]!= None):
            if points_coronal[0,1]>points_coronal[1,1]:
                temp = points_coronal[0,1]
                points_coronal[0,1]=points_coronal[1,1]
                points_coronal[1,1]=temp
            if points_coronal[0,0]>points_coronal[1,0]:
                temp = points_coronal[0,0]
                points_coronal[0,0]=points_coronal[1,0]
                points_coronal[1,0]=temp
        if (points_axial[0,1]!= None or points_axial[1,1]!= None):
            if points_axial[0,1]>points_axial[1,1]:
                temp = points_axial[0,0]
                points_axial[0,0]=points_axial[1,0]
                points_axial[1,0]=temp
            # print(['-']*30)
            # print('Select Axial area')
            # print(['-']*30)
            # points_axial = draw_rect(sitk.GetArrayViewFromImage(CTRL_images[0])[selected_slice_axial,:,:])
            extract = sitk.RegionOfInterestImageFilter()
            extract.SetRegionOfInterest([int(points_coronal[0,1]),int(points_axial[0,0]),int(points_coronal[0,0]),
                                         int(points_coronal[1,1]-points_coronal[0,1]),int(points_axial[1,0]-points_axial[0,0]),int(points_coronal[1,0]-points_coronal[0,0])])
            CTRL_cropped = []
            AD_cropped = [] 
            for x in CTRL_images:
                cropped = extract.Execute(x)
                CTRL_cropped.append(cropped)
            for x in AD_images:
                cropped = extract.Execute(x)
                AD_cropped.append(cropped)
            
            # fig, (ax1, ax2, ax3) = plt.subplots(3)
            # ax1.imshow(sitk.GetArrayViewFromImage(CTRL_cropped[0][:,:,20]))
            # ax2.imshow(sitk.GetArrayViewFromImage(CTRL_cropped[0][:,25,:]))
            # ax3.imshow(sitk.GetArrayViewFromImage(CTRL_cropped[0][20,:,:]))

#%% Select witch image dataset you want to use if you cropped the image       
    if Y_N == 'Yes' and (points_coronal[0,1]!= None or points_coronal[1,1]!= None):
        CTRL_uncropped = CTRL_images
        AD_uncropped = AD_images
        C_U = ''
        while C_U != 'Cropped' and C_U != 'Uncropped':
            C_U = input("\nWhat dataset do you want to use: 'Cropped' or 'Uncropped'?")
            if (C_U == 'Cropped'):
                CTRL_images = CTRL_cropped
                AD_images = AD_cropped
            elif C_U == 'Uncropped':
                break
            else:
                print("\nUnvalid input, please try again with 'Cropped' or 'Uncropped'.")
            
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
#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image
    # import multiprocessing
    # CTRL_GM = []
    # AD_GM = []
    # def Filter_GM(x):
    #     threshold_filters= sitk.LaplacianSharpeningImageFilter()#first selection of the zones
        
    #     thresh_img = threshold_filters.Execute(x)
    
    #     threshold_filters= sitk.UnsharpMaskImageFilter() #enhancement of the edges in order to set a more accurate threshold
    #     thresh_img = threshold_filters.Execute(thresh_img)
    #     #threshold_filters= sitk.YenThresholdImageFilter() #this is a good threshold too but it's a little blurry
    #     threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
    #     threshold_filters.SetInsideValue(0)#
    #     threshold_filters.SetOutsideValue(1)#binomial I/O
    #     thresh_img = threshold_filters.Execute(thresh_img)
    #     data = sitk.GetArrayFromImage(thresh_img)
    #     #Taking GM elements
    #     filtered_img = np.where(data == 1, sitk.GetArrayViewFromImage(x), data)
    #     return filtered_img
    # def CTRL_filtration(x):
    #     global CTRL_GM
    #     filtered_img = Filter_GM(x)
    #     filtered_img = filtered_img[:,:,:]#select slices to append
    #     return filtered_img.flatten()
    # def AD_filtration(x):
    #     global AD_GM
    #     filtered_img = Filter_GM(x)
    #     filtered_img = filtered_img[:,:,:]#select slices to append
    #     AD_GM.append(filtered_img.flatten())    
    # num_cores = multiprocessing.cpu_count()
    # print('N° Cores = {}'.format(num_cores))
    # start = perf_counter()
    # pool = multiprocessing.Pool(processes = num_cores)
    
    # resoults = pool.map(CTRL_filtration, CTRL_images)
    # pool.close() 
    
    # # for x in CTRL_images:
    # #     CTRL_filtration(x)
    # # for x in AD_images:
    # #     AD_filtration(x)
    # print("Time: {}".format(perf_counter()-start))#Print performance time
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
    print(classifier)
    print(classifier.score(test_set_data, test_set_lab))
    #%% # create SVC than extract more relevant feature with selector (weigth^2)
        #Try RFE
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    train_set_data, test_set_data, train_set_lab, test_set_lab = train_test_split(dataset, labels, test_size = 0.3,random_state=42)
    X, y = train_set_data, train_set_lab
    n_splits = 20#secondo articolo(mi sembra) 
    classifier = SVC(kernel='linear', probability=True)
    start = perf_counter()
    rfe = RFE(estimator=classifier, n_features_to_select=6000, step=0.5)#Classic RFE
    #rfe = RFECV(estimator=classifier, min_features_to_select=6000, step=0.5, n_jobs=-1)#Find BEST feature with cross validation, it can be paralelized so it's a little bit faster but the cross validation takes time
    FIT = rfe.fit(X,y)
    print("Time: {}".format(perf_counter()-start))#Print performance time
    fig, ax = plt.subplots()
    arr = sitk.GetArrayViewFromImage(CTRL_images[0])
    ax.imshow(arr[:,int(np.round(arr.shape[1]/2)),:], cmap = 'Greys_r')
    ax.imshow(FIT.support_.reshape(arr.shape[0], arr.shape[1], arr.shape[2]).astype(int)[:,int(np.round(arr.shape[1]/2)),:], alpha = 0.6, cmap='RdGy_r')
    #Base performance with actual dataset
    classifier = classifier.fit(X, y)
    print(classifier.score(test_set_data, test_set_lab))
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
    #Resume
    print("Time: {}".format(perf_counter()-start))
    print(classifier.score(test_X, test_set_lab))

    #%%RFE: Home made, every iteration eliminate the lowest n feature in weigth vector and fit again the model until the target size is reached
    train_set_data, test_set_data, train_set_lab, test_set_lab = train_test_split(dataset, labels, test_size = 0.3,random_state=42)
    data = train_set_data
    t_d, t_t = test_set_data, test_set_lab
    start = perf_counter()
    feature_rank = []
    v_used = [i for i in range(dataset.shape[1])]
    idx = []
    target_size = 6000
    while data.shape[1]>target_size:
        classifier = classifier.fit(data, train_set_lab)
        coef_vect = np.abs(classifier.coef_)
        coef_vect = coef_vect.tolist()[0]
        elim = np.sort(coef_vect)[0:int(np.round(data.shape[1]/2))]
        indx = []
        for i in elim:
            indx.append(coef_vect.index(i))
        data = np.delete(data,indx,1)
        idx.append(indx)
        print("Time: {}".format(perf_counter()-start))#Print performance time
    classifier = classifier.fit(data, train_set_lab)
    coef_vect = classifier.coef_ 
    print(classifier)
    for i in idx:
        t_d = np.delete(t_d,i,1)
    print(classifier.score(t_d, t_t))
    ##mask of best pixel
    rfe_voxels = np.where(dataset[0]== np.any(data[0]), 1,0)
    plt.imshow(np.reshape(rfe_voxels,sitk.GetArrayFromImage(CTRL_images[0]).shape)[:,50,:])
    #%%Its good but cannot be implemen
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    data = dataset
    lsvc = LinearSVC(C=1, penalty="l1", dual=True).fit(train_set_data, train_set_lab)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(train_set_data)
    X_new.shape


    #%%
    rfe = RFE(estimator=SVC(kernel='linear', probability=True), n_features_to_select=6000, step=0.5)
    model = SVC(kernel='linear', probability=True)
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


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