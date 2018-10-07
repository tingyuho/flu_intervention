"""
Created on Sat Oct 6 02:46 2018

@author: TingYu Ho
"""
import numpy as np
import pandas as pd
import os
import pickle

def fitting_model_function(X,r):
    r = 1
    mu, sigma = 0, 0 # mean and standard deviation
    s = np.random.normal(mu, sigma, r)
    x1 = X[0]
    x2 = X[1]

    _DIREPATH_ = os.getcwd()

    fit_file_name = 'subspace_fit_function.csv'
    _RESULT_ = os.path.join(_DIREPATH_, 'results')
    _FIT_FILE_DIR_ = os.path.join(_RESULT_, fit_file_name)

    df_subspaces = pd.read_csv(_FIT_FILE_DIR_)
    df_subspaces_filter = df_subspaces.loc[df_subspaces.activate == True]
    #fit_model_name = str(df_subspaces_filter[[x1 >= df_subspaces_filter.rbmt_min & x2 <= df_subspaces_filter.cs_max]].fit_regressor)+'.sav'
    fit_model_name = 'rf_fit_model_rbmt_min_0_cs_max_1.sav'
    loaded_model = pickle.load(open(os.path.join(_RESULT_, fit_model_name), 'rb'))
    y = loaded_model.predict([[x1, x2]])

    return -y

if __name__ == "__main__":
    print(fitting_model_function([3,0.2], 1))