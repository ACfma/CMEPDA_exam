# CMEPDA_exam

CMEPDA_exam is a repository containing two Python modules capable to classify Alzheimer desease subjects from control's ones using GM segmented MRI images.

The images used in order to create and test these modules can be found in the IMAGES folder.

---
## model_cnn

model_cnn is a Python module that uses a 3D Convolutional Neural Network in order to accomplish the task mentioned above.

model_cnn can be launched by the omonimous .ipynb file (Google Colab Notebook). The other .py files are the documented version of the functions used in the Notebook.


## model_svm

model_svm is a Python module that uses a linear SVM classifier along with RFE and PCA as dimentionality reductors. It is also ment to be used over different kind of dataset (as long as the files are the ones mentioned above).

model_svm can be launched by the omonimous .py file, while the other .py files are used within the main program.

model_svm will also save different maps of the most important features found as .nii files (see example below) depending on the extractor used.

<img src="https://github.com/ACfma/CMEPDA_exam/blob/main/IMAGES/summed_ctrl.png" height="250" width="250">

---

**These modules have been created using Python 3.7.**

***Please, read the documentation (https://readthedocs.org/projects/cmepda-exam/badge/?version=latest) and requirements (requirements.txt) for further details.***

---

[![Documentation Status](https://readthedocs.org/projects/cmepda-exam/badge/?version=latest)](https://cmepda-exam.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ACfma/CMEPDA_exam.svg?branch=main)](https://travis-ci.org/ACfma/CMEPDA_exam)
