#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 12:21:05 2025

@authors:    Yazan Mash'Al (5443768),
            Lars de Hoop (5644690),
            Lucas Verbeeke (5650534)
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 1000


# we simulate two types of features
half_sample=int(n/2)
x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
simulated_features = np.vstack((x1, x2)).astype(np.float64)


# the underlying value of beta in the simulation; the value we want to retrieve in the estimation procedure
beta_star=np.array([0.2,-0.8])



#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# Simulate the labels
def logistic_simulation(features,beta):    
    signal = np.dot(features, beta)
    p=logistic(signal)
    y= np.array([np.random.binomial(1, p[i] ) for i in range(n)])
    return y
 




simulated_labels = logistic_simulation(simulated_features, beta_star)



#### Scatter plot of the features and correspoding labels
plt.figure(figsize=(12,8))
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c = simulated_labels, alpha = .5)
plt.show()




#Skeleton for function Newton-Raphson for logisic regression

def logistic_regression_NR(features, target, num_steps, tolerance):
    beta = np.zeros(features.shape[1])
    for step in range(num_steps):
        p = logistic(np.dot(features, beta))
        gradient = np.dot(features.T, target - p)
        hessian = - np.dot(features.T, p * (1 - p) * features)
        beta -= np.dot(np.linalg.inv(hessian), gradient)
        if np.linalg.norm(gradient) < tolerance:
            break
                
    return beta


## Simulation study
S=1000
#generate labels y for every simulation
#simulated_labels = logistic_simulation(simulated_features, beta_star)
#compute the MLE for every simulation


#compute the means of estimated parameters beta_1 and beta_2
#make a histrogram for the MLE of beta_1 and beta_2

#plt.hist(XX,bins=)







