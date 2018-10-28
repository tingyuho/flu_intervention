import subprocess, sys, os, shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from gplearn.genetic import SymbolicRegressor
from pyDOE import *
from contextlib import contextmanager
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import os
import argparse
ap = argparse.ArgumentParser(description='flute-low-fidelity-model-builder')

ap.add_argument('--rf_depth', type=int)
ap.add_argument('--rf_save', type=str)

argv = ap.parse_args()

from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
rangeReimbursment, rangeCostSharing = [0, 20], [0, 1]

cost_vaccine = 35
weight_infect_base = 40
weight_hospital = 600
weight_dead = 1000
_MAX_RF_DEPTH_= argv.rf_depth
_MODEL_SAVE_ = argv.rf_save

if _MODEL_SAVE_ == 'True':
    _MODEL_SAVE_ = True
else:
    _MODEL_SAVE_ = False
_DIREPATH_ = os.getcwd()
_RESULT_ = os.path.join(_DIREPATH_, 'results')
cost_file_name = 'insurer_cost_summary_case1.csv'
_COST_FILE_DIR_ = os.path.join(_RESULT_, cost_file_name)
df_input_response = pd.read_csv(_COST_FILE_DIR_, index_col=False)
df_input_response = df_input_response.dropna()

#
# df_input_response['Cost_include_vaccine_cost'] = df_input_response.total_intervention_cost + df_input_response.N_vaccination*cost_vaccine
# df_input_response['Cost_include_infection_hopsital_death_cost'] = df_input_response.total_intervention_cost+ \
#                                                                   weight_infect*df_input_response.Cumulated_Infection + \
#                                                                   weight_hospital*df_input_response.N_hospitalized + \
#                                                                   weight_dead*df_input_response.N_death
# df_input_response['Infection_hopsital_death_cost'] = weight_infect*df_input_response.Cumulated_Infection + \
#                                                                   weight_hospital*df_input_response.N_hospitalized + \
#                                                                   weight_dead*df_input_response.N_death
# df_input_response['Total_cost'] = df_input_response.total_intervention_cost+ \
#                                                                   weight_infect*df_input_response.Cumulated_Infection + \
#                                                                   weight_hospital*df_input_response.N_hospitalized + \
#                                                                   weight_dead*df_input_response.N_death+\
#                                                                   df_input_response.N_vaccination*cost_vaccine



X = df_input_response[['reimbursement','cost_sharing']].values.tolist()
X_train = X[0:int(len(X)*0.8)]
X_test = X[int((len(X)*8)/10):len(X)]

#figure_list = ['total_intervention_cost', 'Cumulated_Infection', 'N_vaccination', 'N_hospitalized', 'N_antivirus', 'Cost_include_vaccine_cost', 'Cost_include_infection_hopsital_death_cost', 'Infection_hopsital_death_cost','Total_cost']
figure_list = ['Test_Total_cost']
for i in range(10):
    weight_infect = weight_infect_base * i
    df_input_response['Test_Total_cost'] = df_input_response.total_intervention_cost + \
                                           weight_infect * df_input_response.Cumulated_Infection

    for figure in figure_list:

        y_total = df_input_response[figure].values.tolist()
        y_train = y_total[0:int(len(y_total)*0.8)]
        y_test = y_total[int((len(y_total)*8)/10):len(y_total)]

        regr = RandomForestRegressor(max_depth=_MAX_RF_DEPTH_, random_state=0, n_estimators=100)
        regr.fit(X_train, y_train)

        if _MODEL_SAVE_ is True:
            filename = figure+'_weight_'+str(weight_infect) + '.sav'
            pickle.dump(regr, open(os.path.join(_RESULT_, filename), 'wb'))

        x1 = np.arange(rangeReimbursment[0], rangeReimbursment[1], (rangeReimbursment[1]-rangeReimbursment[0]) / 50.)
        x2 = np.arange(rangeCostSharing[0], rangeCostSharing[1], (rangeCostSharing[1]-rangeCostSharing[0]) / 50.)
        x1, x2 = np.meshgrid(x1, x2)
        y_regr = regr.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

        #if figure == 'total_intervention_cost':
        ax = plt.figure().gca(projection='3d')
        ax.set_xlim(rangeReimbursment[0], rangeReimbursment[1])
        ax.set_ylim(rangeCostSharing[0], rangeCostSharing[1])
        ax.set_xlabel('reimbursement', labelpad = 9, fontsize=20)
        ax.set_ylabel('cost sharing rate', labelpad = 9, fontsize=20)
        ax.set_zlabel(figure+'_'+str(weight_infect), labelpad = 9, fontsize=20)
        surf = ax.plot_surface(x1, x2, y_regr, rstride=4, cstride=4, cmap='Blues_r')
        points = ax.scatter([X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))], y_train, c='green', s=30)

    plt.show()



