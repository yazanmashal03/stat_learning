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
import pandas as pd
import matplotlib.pyplot as plt

#read the csv file
df = pd.read_csv(r'mnist.csv')
# Alternativelyyou can put the file in your working directory
# If you load the csv file with another function make sure that the matrix of features X is defined as in the book
# and the assignment and convert it to an numpy array

#make dataframe into numpy array
df.to_numpy()
y_labels_data=df['label'].to_numpy()
df_xdata=df.drop(columns='label')
x_features_data=df_xdata.to_numpy()
x_features_data.shape

# we will only use the zeros and ones in this empirical study
y_labels_01 = y_labels_data[np.where(y_labels_data <=1)[0]]
x_features_01 = x_features_data[np.where(y_labels_data <=1)[0]]

# create training set
n_train=100
y_train=y_labels_01 [0:n_train]
x_train=x_features_01[0:n_train]

#create test set
n_total=y_labels_01.size
y_test=y_labels_01 [n_train:n_total]
x_test=x_features_01[n_train:n_total]

## Here we plot some handwritten digits

plt.figure(figsize=(25,5))
for index, (image, label) in enumerate(zip(x_train[5:10], y_train[5:10])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Label: %i\n' % label, fontsize = 20)

#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))

#given an estimated parameter for the logistic model we can obtain forecasts 
#for our test set with the following function

def logistic_forecast(features,beta):    
    signal_hat = np.dot(features, beta)
    y_hat=np.sign(signal_hat)
    y_hat[y_hat<0] = 0
    return y_hat
  
#computes the prediction error by comparing the predicted and the observed labels in the test set
def prediction_accuracy(y_predicted,y_observed):
    errors= np.abs(y_predicted-y_observed)
    total_errors=sum(errors)
    acc=1-total_errors/len(y_predicted)    
    return acc

#dimension of the problem
p=x_train.shape[1]

#Compute the ranks of the matrices X and X^T
rank_x_train = np.linalg.matrix_rank(x_train)
print(f"Rank of x_train: {rank_x_train} out of {x_train.shape[1]} features")
rank_x_trans = np.linalg.matrix_rank(x_train.T)
print(f"Rank of x_train_trans: {rank_x_trans} out of {x_train.T.shape[1]} features")


def logistic_regression_NR(features, target, num_steps=100, tolerance=1e-6):
    beta = np.zeros(features.shape[1])
    for step in range(num_steps):
        p = logistic(np.dot(features, beta))
        W = p * (1 - p)
        gradient = np.dot(features.T, target - p)
        hessian = - np.dot(features.T, W[:, np.newaxis] * features)
        beta -= np.dot(np.linalg.inv(hessian), gradient)
        if np.linalg.norm(gradient) < tolerance:
            break
                
    return beta

# Regularisation parameter
lambda_0=1


def logistic_regression_NR_penalized(features, target, num_steps=100, tolerance= 1e-6):
    beta = np.zeros(features.shape[1])
    for step in range(num_steps):
        p = logistic(np.dot(features, beta))
        W = p * (1 - p)
        gradient = np.dot(features.T, target - p) - lambda_0 * beta
        hessian = - np.dot(features.T, W[:, np.newaxis] * features) - lambda_0 * np.eye(features.shape[1])
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            print("Still singular, even after regularization.")
            return None
        beta -= delta
        if np.linalg.norm(gradient) < tolerance:
            break
                
    return beta

beta_pen = logistic_regression_NR_penalized(x_train, y_train)
print('This is the beta_pen:', beta_pen)
beta_unpen = logistic_regression_NR(x_train, y_train)
