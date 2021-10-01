# -*- coding: utf-8 -*-

#%% Import libraries
import numpy as np
import spotpy 
#import chaospy as cp
import configparser
import pandas as pd
import random
import math
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GausianLike
from spotpy.analyser import get_best_parameterset, plot_parameter_trace, get_modelruns
from spotpy.analyser import plot_posterior_parameter_histogram
import matplotlib.pyplot as plt

#%% Setup model class (parameters, simulation, evaluation and likelihood)
K_skalar = 8.64 #[m/d]
b = 10 #saturated thickness [m]
sy = 0.25 #specific yield

def calculate_hxt_for_one_variable(D_input, A_input, w_input, x_input, t_input):
  fun1 = A_input * math.exp(-x_input * math.sqrt(w_input / (2*D_input)))
  fun2 = math.sin(-x_input * math.sqrt((w_input / (2*D_input)) + ((w_input * t_input) )))
  return fun1 * fun2
#%%
class spot_setup(object):
    def __init__(self, _used_algorithm):
        
        self._used_algorithm = _used_algorithm
        self.params = [spotpy.parameter.Uniform('logA', low= -1, high=1),
                       spotpy.parameter.Uniform('logK', low=0, high=2),
                       spotpy.parameter.Uniform('w', low= (2*math.pi)/7, high=(2*math.pi)/2)]

#[spotpy.parameter.Uniform('logA', low= math.log(0.01, 5), high=math.log(1, 5)) 
#spotpy.parameter.Uniform('logK', low=np.log10(0.01), high=np.log10(2))                      
#%%   Define uncertain parameters         
    def parameters(self):
        return spotpy.parameter.generate(self.params)
#%% Run simulations
    def simulation(self, x):
        
        logA, logK, w = x[0], x[1], x[2]
        phi = 0
        
        A = 5 ** logA
        
        D = (10 ** logK) * 8640 * b / sy
        
        times = [7, 10, 25, 50, 180] #days
        distances = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] #m
        
        simulation = np.zeros((len(times), len(distances)))
        
        for t , _ in enumerate(times):
           for x , _ in enumerate(distances):
               simulation[t][x] = calculate_hxt_for_one_variable(D, A, w, x, t)
               
        return simulation.flatten()
        
    
#%% Import observed values
    def evaluation(self):
        
        obs = np.zeros((5,10))
        obs[0,:] =  [-0.102488214832666, -0.213669206729949, -0.243068241621553	,	-0.0644217683867324	,	-0.0603414471556082	]
        obs[1,:] =  [0.443999406954099	, 0.303188299476753 ,	0.399381989561914	,	0.181871866404912	,	-0.171663585230874	]
        obs[2,:] =  [-0.371461399773691	,	0.0366151360819552	,	0.306417078570019	,	0.192870096397016	,	0.183152310977500	]
        obs[3,:] =  [0.782927857935940,		0.113570122237571	,	-0.160172200979472,		-0.00387621599071414	,	-0.00207478866324770	]
        obs[4,:] =  [-0.976539641786550,		-0.308098743266651,		-0.00877293988316697	,	0.167401153463919	,	 0.0780282510863342]  
       
        return obs.flatten()
    
#%% Define likelihood function

    def objectivefunction(self, simulation, evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation, simulation)
        return like

