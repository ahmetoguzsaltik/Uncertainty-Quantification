#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%import libraries

#pip install spotpy

import numpy as np
import spotpy 
from dream_setup import spot_setup
from spotpy.analyser import get_best_parameterset


#%% dream_run
if __name__ == "__main__":
    
    #set the random state
    random_state = 321
    np.random.seed(random_state)
    
    #general settings
    parallel = 'seq'
    spot_setup = spot_setup(_used_algorithm='dream')
    
    #select number of maximum repetitions
    rep = 10000
    
    #number of chains and convergence limit (Gelman_Rubin)
    nChains = 6
    convergence_limit = 1.2
    
    # define DREAM algorithm parameters (further details in Vrugt, 2016)
    nCr = 3
    eps = 10e-6
    runs_after_convergence = 1000
    acceptance_test_option = 6
    
    # initiate DREAM algorthm
    sampler = spotpy.algorithms.dream(spot_setup,
                                       dbname ='dream_fluctHeads_4',
                                       dbformat = 'csv',
                                       db_precision = np.float32,
                                       save_sim = True,
                                       random_state=random_state)

    r_hat = sampler.sample(rep,
                       nChains = nChains,
                       convergence_limit = convergence_limit,
                       runs_after_convergence = runs_after_convergence)
    
    #load the likelihood, and the parameter values of all simulations
    results = sampler.getdata()
    
    
    

    # load the likelihood, and the parameter values of all simulations
    results = sampler.getdata()
    best_parameterset = get_best_parameterset(results, maximize=True)
    #get_modelruns(results)
    
    #Re-convert from log scale to normal scale
    A = 10 ** best_parameterset[0][0] 
    D = (10 ** best_parameterset[0][1]) * 8640 / 0.25
    w = best_parameterset[0][2]
    
    print("Best parameter set: A=" + str(A) + " D=" + str(D) + " omega=" + str(w))
