# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:32:01 2021

@author: acker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import math

    
#%%
distances = np.linspace(1, 100, 100)
times = [7, 30, 180]
dist = [10, 20, 50]
numSamples = 100

# deterministic parameters
phi = 0
n = 0.25 #porosity
K_skalar = 8640 #skalar of hydraulic conductivity

dx=0.0001#m


#we use the by the DREAM Algorithm calculated combinations of assignment 4:
parameters = pd.read_csv('dream_fluctHeads_together.csv')

A = np.zeros(100)
K = np.zeros(100)
K_multi = np.zeros((numSamples, len(times), len(dist)))
omega = np.zeros(100)





for i in range(100):
    K[i] = K_skalar* 10** parameters.parlogK[i] #[m/d]
    
    K_multi[i] = K_skalar* 10** parameters.parlogK[i] #[m/d]
    
    A[i] = 5**parameters.parlogA[i]#[m]
    
    omega[i] = parameters.parw[i] #[d]
    


#%%    

b = 10 #saturated thickness [m]
sy = 0.25 #specific yield

def calculate_hxt_for_one_variable(D_input, A_input, w_input, x_input, t_input):
  fun1 = A_input * math.exp(-x_input * math.sqrt(w_input / (2*D_input)))
  fun2 = math.sin(-x_input * math.sqrt((w_input / (2*D_input)) + ((w_input * t_input) )))
  return fun1 * fun2

#%%

#calculate the dispersivity using the K values of assignment 4
D = (K[:] * b) / sy  


# calculate hydraulic head for the model domain and the time steps
head = np.zeros((numSamples, len(times), len(dist)))
head_1 = np.zeros((numSamples, len(times), len(dist)))
v = np.zeros((numSamples, len(times), len(dist)))

#%% calculate the groundwater velocities with the 100 parameter combinations for all combinations of x and t

for i in range(numSamples):
    parA = A[i]
    parD = D[i]
    parOmega = omega[i]
    
    for t, T in enumerate(times):
        for x,  X in enumerate(dist):
            head[i][t][x] = calculate_hxt_for_one_variable(parD, parA, parOmega, dist[x], times[t])
            x_1 = X + dx
            head_1[i][t][x] = calculate_hxt_for_one_variable(parD, parA, parOmega, x_1, times[t])
            
            #calculate the groundwater velocity using Darcy's Law
            v =  -(K_multi/n)*(head_1 - head) / dx
            
            
#%% plot a histogram of the calculated velocities for all combinations of x and t
for i, t in enumerate(times):
    for j, x in enumerate(dist):
        
        z, bins, patches = plt.hist(x=v[:,i,j], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        #plt.xscale('log')
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('velocity [m/d]')
        plt.ylabel('Frequency')
        plt.title('Histogram(x='+str(x)+'m, t='+str(t)+'h)')
        maxfreq = z.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()
        
#%% calculation and plotting of the arrival times at 10m,20m and 50m away from the shoreline (at x=1m) 
#for computational reasons we seperate the calculation of the arrival time at x=50m


dt=0.0004#days
dx=0.0001#m
dist = [10,20,50]
x_results = []
t_results = []
#%% 
for e, E in enumerate(dist):
    t_results_plot = []
    for i,_ in enumerate(range(100)):
        
        parK = K[i]
        parD = D[i]
        parOmega = omega[i]
        
        x_cur = 0
        x = 1
        t = 0
        
        
        
        velocity = []
        
        while x < E:
            
            head_cur = calculate_hxt_for_one_variable(parD,parA,parOmega,x,t)
            
            x_1 = x + dx
            
            head_cur_1 = calculate_hxt_for_one_variable(parD,parA,parOmega,x_1,t)
            
            v_cur = - parK/n * (head_cur_1 - head_cur)/dx
            velocity.append(v_cur)
            x = x + v_cur * dt
            t = t + dt
            
        t_results.append(t)
            
        x_results.append(x)
        
        t_results_plot.append(t)
        
    z, bins, patches = plt.hist(x=t_results_plot[:], bins='auto', color='#0504aa',
                        alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('time [d]')
    plt.ylabel('Frequency')
    plt.title('Histogram(x='+str(E)+'m)')
    maxfreq = z.max()
    #Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


