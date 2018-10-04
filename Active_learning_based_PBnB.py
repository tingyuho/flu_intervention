import subprocess, sys, os, shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from gplearn.genetic import SymbolicRegressor
from pyDOE import *
from contextlib import contextmanager
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

@contextmanager  # with suppress_stdout():
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def flu_simulation(name_config): # running flu simulation in cmd
    print("running flu_simulation:  " + name_config)
    #FNULL = open(os.devnull, 'w')
    #p = subprocess.Popen(r'flute '+name_config, cwd=r'C:\Users\TingYu Ho\PycharmProjects\SybolicRegression\gplearn_sr', stdout=FNULL, stderr=subprocess.STDOUT)
    p = subprocess.Popen(r'flute ' + name_config,
                         cwd=r'C:\Users\TingYu Ho\PycharmProjects\SybolicRegression\gplearn_sr')
    # p = subprocess.Popen(r'flute config-twodose', cwd=r'C:\Users\TingYu Ho\Google Drive\Research paper\Flu\FluTE-origin')
    # Wait for calling function to terminate
    p_status = p.wait()



def multiprocess(list_configs): # implement parallel multiprocess
    pool = mp.Pool(processes=4)
    [pool.apply_async(
            func=flu_simulation,
            args=(config,)
        ) for config in list_configs]
    pool.close()
    pool.join()


if __name__ == "__main__":

    # remove existing file
    if os.path.exists("insurer_cost_summary.csv"): os.remove("insurer_cost_summary.csv")
    if os.path.exists("subspace_fit_function.csv"): os.remove("subspace_fit_function.csv")

    # setting
    Nsampling = 100 # number of sampling points for each subregion
    rangeReimbursment, rangeCostSharing = [0, 20], [0, 0.5]
    Partition_R_threshold = 0.99
    nReimbursementPartition = nCostSharingPartition = 0
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']
    conf_origin = 'config-Seattle-origin-root'
    config_origin_test = 'config-highrisk'

    # initialize dataframe of subspaces
    column_names = ['rbmt_min','rbmt_max','cs_min','cs_max','Nsmapling','activate',\
                    'complete_learning','fit_function','cod','fit_regressor',]
    df_subspaces = pd.DataFrame(data = [[rangeReimbursment[0], rangeReimbursment[1], rangeCostSharing[0], rangeCostSharing[1],\
                                       0, True, False, "", None, None]], index=[1], columns=column_names)



    while True:
        """
        step 1: DOE to create the N sampling point
        """
        print ("step 1: DOE to create the N sampling point")
        for ind, row in df_subspaces.iterrows(): # for each subspace
            if (row.complete_learning is False) & (row.activate is True):
                N_lhs_generator = Nsampling - row.Nsmapling # number of newly generate points

                # TODO: add augmented lhs
                print("N_lhs_generator: "+str(N_lhs_generator))
                if N_lhs_generator == 1: N_lhs_generator += 1
                df_subspaces.loc[ind, 'Nsmapling'] = row['Nsmapling']+N_lhs_generator  # update the number of sampling
                if N_lhs_generator > 0:
                    x = lhs(2, samples=N_lhs_generator, criterion='maximin') # [0,1]
                    # reshape the random points

                    x[0:N_lhs_generator, 0] = x[0:N_lhs_generator, 0] * (row.rbmt_max-row.rbmt_min) \
                                              + row.rbmt_min
                    #x[0:N_lhs_generator, 0] = [round(i, 4) for i in x[0:N_lhs_generator, 0]]
                    x[0:N_lhs_generator, 1] = x[0:N_lhs_generator, 1] * (row.cs_max-row.cs_min) \
                                              + row.cs_min
                    #x[0:N_lhs_generator, 1] = [round(i, 4) for i in x[0:N_lhs_generator, 1]]

                    """
                    step 2: Create the N_lhs_generator configure files
                    """
                    print("step 2: Create the N_lhs_generator configure files")
                    list_configure = []

                    for point in x:


                        conf_name = 'config-Seattle-'+str(point[0])+'-'+str(point[1])+'_file'
                        f = open(conf_name, "w+")
                        f.close()

                        shutil.copy(conf_origin, conf_name)

                        # insert tje configuration setting
                        f = open(conf_name, "a+")

                        #f.write("label " + "example-minimal\n")
                        #f.write("datafile one\n")
                        """
                        f.write("R0 %.2f\r" % 1.6)
                        f.write("seed %d\r" % 1)
                        f.write("seedinfected %d" % 10)
                        
                        f.write("seedinfecteddaily %d\r" % 0)
                        """
                        f.write("reimbursement %.2f\n" % point[0])
                        f.write("costsharingrate %.10f" % point[1])

                        f.close()
                        list_configure.append(conf_name)

                    """
                    step 3: Parallel computing Flute and write to csv file   
                    EX: list_config = ['configure_1','configure_2']
                    """
                    print("step 3: Parallel computing Flute and write to csv file")

                    multiprocess(list_configure)

        """
        step 4: Read the csv file, apply active learning, and get the symbolic Symbolic regression function 
        """
        print("step 4: Read the csv file, apply active learning, and get the symbolic Symbolic regression function ")
        file = "insurer_cost_summary.csv"
        df_input_response = pd.read_csv(file, index_col=False)
        print(df_input_response.columns.values)

        #df_input_response['total_intervention_cost'] = df_input_response['total_intervention_cost'].apply(lambda x: round(x/float(1000000000),6))

        # for the subspace whose complete learning is false"
        for ind, row in df_subspaces.iterrows():
                # get the training data
            if (row.complete_learning is False) & (row.activate is True):
                df_input_response_train_testing = df_input_response[(df_input_response.reimbursement >= row.rbmt_min) & \
                                                            (df_input_response.reimbursement <= row.rbmt_max) & \
                                                            (df_input_response.cost_sharing >= row.cs_min) & \
                                                            (df_input_response.cost_sharing <= row.cs_max)]
                X = df_input_response_train_testing[['reimbursement','cost_sharing']].values.tolist()
                y = df_input_response_train_testing['total_intervention_cost'].values.tolist()

                X_train = X[0:int(len(X)*0.8)]
                X_test = X[int((len(X)*8)/10):len(X)]

                y_train = y[0:int(len(y)*0.8)]
                y_test = y[int((len(y)*8)/10):len(y)]

                #min_max_scaler = preprocessing.MinMaxScaler()

                #x_scaled = min_max_scaler.fit_transform(x)
                #y_scaled = min_max_scaler.fit_transform(y)

                #df_input_response_train_normalized_x = pd.DataFrame(x_scaled, columns=['reimbursement','cost_sharing'])
                #df_input_response_train_normalized_y = pd.DataFrame(y_scaled, columns=['total_intervention_cost'])

                #X_processed = df_input_response_train_normalized_x[['reimbursement','cost_sharing']].values.tolist()
                #y_processed = df_input_response_train['total_intervention_cost'].values.tolist()
                print('X_train:')
                print(X_train)
                print('y_train')
                print(y_train)

                """
                # random forest prediction
                """
                regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=100)
                regr.fit(X_train, y_train)

                """
                # symbolic regressor 
                est_gp = SymbolicRegressor(population_size=80000,
                                           generations=5, stopping_criteria=0.001, const_range=(-100000, 100000),
                                           p_crossover=0.7, p_subtree_mutation=0.1,
                                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                                           max_samples=0.9, verbose=1, function_set=function_set,
                                           parsimony_coefficient='auto', random_state=0,  metric='mean absolute error')
                print('begin_training')
                est_gp.fit(X_train, y_train)
                """
                df_subspaces.loc[ind, 'cod'] = str(regr.score(X_train, y_train))
                print ("trainging score: "+str(regr.score(X_train, y_train)))
                print ("testing score: " + str(regr.score(X_test, y_test)))
                """
                plot
                """

                x1 = np.arange(row.rbmt_min, row.rbmt_max, (row.rbmt_max-row.rbmt_min) / 50.)
                x2 = np.arange(row.cs_min, row.cs_max, (row.cs_max-row.cs_min) / 50.)
                x1, x2 = np.meshgrid(x1, x2)
                y_regr = regr.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

                ax = plt.figure().gca(projection='3d')
                ax.set_xlim(row.rbmt_min, row.rbmt_max)
                ax.set_ylim(row.cs_min, row.cs_max)
                surf = ax.plot_surface(x1, x2, y_regr, rstride=1, cstride=1, color='green', alpha=0.5)
                points = ax.scatter([X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))], y_train)
                plt.show()



                if regr.score(X_test, y_test)>= Partition_R_threshold: # output the Coefficient of determination to evaluate the prediction model
                    df_subspaces.loc[ind, 'complete_learning'] = True
                    #df_subspaces.loc[ind, 'fit_function'] = est_gp._program
                    #df_subspaces.loc[ind, 'fit_regressor'] = regr
                    print('complete_training')
        df_subspaces.reindex
        df_subspaces.to_csv('subspace_fit_function.csv', index=False)
        """
        step 5: partition subspace if error rate is too high
        """
        print("step 5: partition subspace if error rate is too high")
        list_newSubspaces = []
        for ind, row in df_subspaces.iterrows():
            if (row.complete_learning is False) & (row.activate is True):
                if nReimbursementPartition <= nCostSharingPartition: # reimbursement axis partition
                    Reimbursement_cent = row.rbmt_min + (row.rbmt_max-row.rbmt_min)/float(2)
                    N_sampling1 = df_input_response[(df_input_response.reimbursement >= row.rbmt_min) & \
                                                            (df_input_response.reimbursement <= Reimbursement_cent) & \
                                                            (df_input_response.cost_sharing >= row.cs_min) & \
                                                            (df_input_response.cost_sharing <= row.cs_max)].shape[0]

                    N_sampling2 = df_input_response[(df_input_response.reimbursement >= Reimbursement_cent) & \
                                                            (df_input_response.reimbursement <= row.rbmt_max) & \
                                                            (df_input_response.cost_sharing >= row.cs_min) & \
                                                            (df_input_response.cost_sharing <= row.cs_max)].shape[0]

                    list_newSubspaces.extend([[row.rbmt_min, Reimbursement_cent, row.cs_min, \
                                               row.cs_max, N_sampling1, True, False, "", None, None], \
                                            [Reimbursement_cent, row.rbmt_max, row.cs_min, \
                                             row.cs_max, N_sampling2, True, False, "", None, None]])
                else: # costsharing axis partition
                    CostSharing_cent = row.cs_min + (row.cs_max - row.cs_min)/float(2)
                    N_sampling1 = df_input_response[(df_input_response.reimbursement >= row.rbmt_min) & \
                                                            (df_input_response.reimbursement <= row.rbmt_max) & \
                                                            (df_input_response.cost_sharing >= row.cs_min) & \
                                                            (df_input_response.cost_sharing <= CostSharing_cent)].shape[0]
                    N_sampling2 = df_input_response[(df_input_response.reimbursement >= row.rbmt_min) & \
                                                            (df_input_response.reimbursement <= row.rbmt_max) & \
                                                            (df_input_response.cost_sharing >= CostSharing_cent) & \
                                                            (df_input_response.cost_sharing <= row.cs_max)].shape[0]
                    list_newSubspaces.extend([[row.rbmt_min, row.rbmt_max, row.cs_min,
                                               CostSharing_cent, N_sampling1, True, False, "", None, None], \
                                              [row.rbmt_min, row.rbmt_max, CostSharing_cent,
                                               row.cs_max, N_sampling2, True, False, "", None, None]])

                df_subspaces.loc[ind, 'activate'] = False
        if nReimbursementPartition <= nCostSharingPartition:  # reimbursement axis partition
            nReimbursementPartition += 1
        else:
            nCostSharingPartition += 1
        if len(list_newSubspaces) > 0:
            df_temp = pd.DataFrame(list_newSubspaces, columns=column_names) # create a temporary dataframe with the new partitioned subspaces
            df_subspaces = df_subspaces.append(df_temp, ignore_index=True)
            df_subspaces.reindex
            print(df_subspaces)
        df_subspaces.reindex
        """
        step 6: Stopping criteria for active learning
        """

        if len(list_newSubspaces) == 0:
            break

    """
    step 7: PBnB
    """