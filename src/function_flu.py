"""
Created on Sat Oct 8 2018
This function is used to feed flute (High-fidelity) model into PBnB
@author: TingYu Ho
"""
import numpy as np
import pandas as pd
import os
import shutil
import subprocess

def function_flu(X,r):
    mu, sigma = 0, 0 # mean and standard deviation
    s = np.random.normal(mu, sigma, r)
    x1 = X[0] # reimbursement
    x2 = X[1] # cost-sharing rate

    _DIREPATH_ = os.getcwd()
    _DATA_ = os.path.join(_DIREPATH_, 'raw_data_flu')

    conf_origin_name = 'config-Seattle-origin-root'
    config_origin_test_name = 'config-highrisk'
    conf_name = 'config-Seattle-' + str(x1) + '-' + str(x2) + '_file'
    cost_file_name = 'insurer_cost_summary.csv'

    if os.path.exists(os.path.join(_DATA_, cost_file_name)): os.remove(os.path.join(_DATA_, cost_file_name))

    _CONFIG_ORIGIN_FILE_DIR = os.path.join(_DATA_, config_origin_test_name)
    _COST_FILE_NAME_DIR_ = os.path.join(_DATA_, conf_name)
    f = open(_COST_FILE_NAME_DIR_, "w+")
    f.close()

    shutil.copy(_CONFIG_ORIGIN_FILE_DIR, _COST_FILE_NAME_DIR_)


    f = open(_COST_FILE_NAME_DIR_, "a+")

    f.write("reimbursement %.2f\n" % x1)
    f.write("costsharingrate %.10f" % x2)

    f.close()

    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(r'flute '+conf_name, cwd=_DATA_, stdout=FNULL, stderr=subprocess.STDOUT)

    # Wait for calling function to terminate
    p_status = p.wait()

    # read cost-input response
    df_input_response = pd.read_csv(os.path.join(_DATA_, cost_file_name), index_col=False)
    noise_funtion = float(df_input_response[-1:].total_intervention_cost)
    return noise_funtion