# CMEPDA_exam

CMEPDA_exam is a repository containing two Python modules capable to classify Alzheimer desease subjects from control's ones using GM segmented MRI images using a single training phase.


---

## split_folder

split_folder is a Python module capable to split a whole images dataset in two folder: [train_set](https://github.com/ACfma/CMEPDA_exam/tree/main/IMAGES/train_set) and [test_set](https://github.com/ACfma/CMEPDA_exam/tree/main/IMAGES/test_set).

These two folders can been used to feed the models below.

## model_cnn

model_cnn is a Python module that uses a 3D Convolutional Neural Network in order to accomplish the task mentioned above using the whole useful image (a parallelepiped surrounding the brain).

model_cnn can be launched by the omonimous .ipynb file (Google Colab Notebook). The other .py files are the documented version of the functions used in the Notebook.

## model_svm

model_svm is a Python module that uses a linear SVM classifier along with RFE and PCA as dimentionality reductors. It may also be used over different kind of dataset (as long as the files are GM segmented MRI images).

model_svm can be launched by the omonimous .py file, while the other .py files are used within the main program.

model_svm will also save different maps of the most important features found as .nii files (see example below) depending on the extractor used.

<img src="https://github.com/ACfma/CMEPDA_exam/blob/main/IMAGES/summed_ctrl.png" height="250" width="250">

(Image created using [Mango®](http://ric.uthscsa.edu/mango/mango.html) )

---

**These modules have been created using Python 3.7.**

***Please, read the documentation (badge below) and requirements ([requirements](https://github.com/ACfma/CMEPDA_exam/blob/main/requirements.txt)) for further details.***

---

[![Documentation Status](https://readthedocs.org/projects/cmepda-exam/badge/?version=latest)](https://cmepda-exam.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ACfma/CMEPDA_exam.svg?branch=main)](https://travis-ci.org/ACfma/CMEPDA_exam)
