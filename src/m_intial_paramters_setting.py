# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 22:15:00 2017

@author: TingYu Ho
"""


import numpy as np
f_delta = 0.1
f_alpha = 0.1

i_k_b = 2
i_B = 2
i_c = 20
i_replication = 2
l_coordinate_lower =  [0, 0] #[-2]
l_coordinate_upper =  [20, 1] #[5]
i_dim = 2 # dimentsion of the domain
D = max((np.array(l_coordinate_upper)-np.array(l_coordinate_lower)).tolist())
f_epsilon = 0.0025*pow(D,i_dim)  # can refer to the Hao's code: fun_diff(l_regionBound)
i_stopping_maxK = 20 # maximum allowable iteration

