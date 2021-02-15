# -*- coding: utf-8 -*-
"""
Tools to compare MMSE from AD_CTRL metadata to CNN prediction


"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

def plot_mmse_prediction(table, test_names, classifier, test_x):
    '''
    Helps to understand how to convert names in the table df with those
    of the split "test_names" list in order to take MMSE and compares them.

    Parameters
    ----------
    test_names : list
        list with test names.
    X : class 'np.array'
        Array of test data.
    classifier : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.

    Returns
    -------
    spr_rank : class 'scipy.stats.stats.SpearmanrResult'
        Sperman Rank.

    '''
    df = pd.read_table(table)
    mmse = []
    data = []
    for j in test_names:
        for i,k in enumerate(df['ID'].values):
            if k + '.' in j:
                mmse.append(df['MMSE'].values[i])
                if 'AD' in k:
                    data.append(True)
                else:
                    data.append(False)

    data = np.array(data)
    mmse = np.array(mmse)
    distances = classifier.predict(test_x)
    plt.figure()
    plt.scatter(mmse[data == True], distances[:,1][data == True],\
                facecolors='none', edgecolors='g')
    plt.scatter(mmse[data == False], distances[:,1][data == False],\
                facecolors='none', edgecolors='b')
    plt.xlabel('MMSE')
    plt.ylabel('Model predict')
    plt.title('Distribution of subject')
    plt.legend(['AD', 'CTRL'],loc="upper left")
    plt.grid()

    spr_rank = spearmanr(mmse,distances[:,1])
    return spr_rank

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Tool to create the model\
                                     and train the dataset")
    parser.add_argument('-table', help='Path to your metadata.csv table', type=str)
    args = parser.parse_args()
    table = args.table
