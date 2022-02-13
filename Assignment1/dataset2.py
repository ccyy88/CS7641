import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTreeClf
from NeuralNetworks import NeuralNetworkClf
from Boosting import BoostingClf
from SVM import SVMClf
from KNN import KNNClf
from preprocessing import *

# filename = 'abalone.csv'
# filename = 'winequality-white.csv'
# encode_col=[0]
# filename = 'online_shoppers_intention.csv'
# encode_col=[10,15,16,17]
filename = 'ObesityDataSet_raw_and_data_sinthetic.csv'
unused_col=[]
encode_col=[0,4,5,8,9,11,14,15,16]
# filename = 'Shill Bidding Dataset.csv'
# unused_col=[0,1,2]
# unused_col=[0]
filepath= './datasets/'

test_size = 0.2
x_train, x_test, y_train, y_test = preprossingdata(filename, filepath, test_size=test_size, unused_col=unused_col, encode_col=encode_col)


# Decision Tree Classifier
dtc = DecisionTreeClf(verbose = True)
# Grid Search
# dtc.gridsearch(x_train, y_train, param_dict,10)
# Plot Learning Curve
dtc.plt_learning_curve(x=x_train, y=y_train, train_sizes = np.linspace(0.01,0.99,10))
# Plot Validation Curve
depth_range = range(1,15)
dtc.plt_validation_curve(x_train,y_train,'max_depth', depth_range)
nodes_range = range(1,60,3)
dtc.plt_validation_curve(x_train,y_train,'max_leaf_nodes', nodes_range)

dtc.evaluate(x_train, y_train, x_test, y_test, max_depth = 10, max_leaf_nodes = 50)




# Neural Network Classifier
nnc = NeuralNetworkClf(verbose = False)
# Grid Search
# nnc.gridsearch(x_train, y_train,5)
# Plot Learning Curve
nnc.plt_learning_curve(x=x_train, y=y_train)
# Plot Validation Curve
# layer_range = [(10,10),(20,20),(50,50),(10,10,10),(20,20,20),(50,50,50)]
layer_range = [(4,6),(4,10),(8,4),(10,5),(8,4,6),(9,6,7),(10,7,8),(20,10),(50,30),(40,80,60)]

nnc.plt_validation_curve(x_train,y_train,'hidden_layer_sizes', layer_range)
batchsize_range = range(10,300,20)
nnc.plt_validation_curve(x_train,y_train,'batch_size', batchsize_range)
# iter_range = range(1,30)
# nnc.plt_validation_curve(x_train,y_train,'max_iter', iter_range)

nnc.evaluate(x_train, y_train, x_test, y_test, hidden_layer_sizes=(10,7,8), max_iter=300)
# {'activation': 'relu', 'hidden_layer_sizes': (4,), 'learning_rate': 'constant', 'solver': 'lbfgs'}
# MLPClassifier(hidden_layer_sizes=(4,), max_iter=300, solver='lbfgs')


# AdaBoosting Classifier
bc = BoostingClf(verbose = False)
# Grid Search
# bc.gridsearch(x_train, y_train,3)
# Plot Learning Curve
base_learner = DecisionTreeClassifier(max_depth=3)
bc.plt_learning_curve(x=x_train, y=y_train, base_learner = base_learner)
# Plot Validation Curve
estimators_range = range(1,1000,50)
bc.plt_validation_curve(x_train,y_train,'n_estimators', estimators_range,base_learner)
learning_rate_range = np.linspace(0.1,1,10)
bc.plt_validation_curve(x_train,y_train,'learning_rate', learning_rate_range, base_learner)

bc.evaluate(x_train, y_train, x_test, y_test, n_estimators=600, learning_rate=1)

#
# SVM Classifier
svmc = SVMClf(verbose = False)
# Grid Search
# svmc.gridsearch(x_train, y_train,3)
# Plot Learning Curve
svmc.plt_learning_curve(x=x_train, y=y_train)
# Plot Validation Curve
kernel_type=['linear', 'poly', 'rbf', 'sigmoid']
svmc.plt_validation_curve(x_train,y_train,'kernel', kernel_type)
c_range = [0.1, 0.2, 0.5,0.8, 1, 2, 5, 8, 10, 20]
svmc.plt_validation_curve(x_train,y_train,'C', c_range)
# gamma_range = [0.1, 0.2, 0.5,0.8, 1, 2, 5, 8, 10, 20]
# svmc.plt_validation_curve(x_train,y_train,'gamma', gamma_range)
svmc.evaluate(x_train, y_train, x_test, y_test, kernel='linear', C=8)

# KNN Classifier
knnc = KNNClf(verbose = False)
# Grid Search
# knnc.gridsearch(x_train, y_train,3)
# Plot Learning Curve
knnc.plt_learning_curve(x=x_train, y=y_train)
# Plot Validation Curve
neighbor_range= range(1,40,2)
knnc.plt_validation_curve(x_train,y_train,'n_neighbors', neighbor_range)
# metric_range = ['minkowski','euclidean','manhattan']
# knnc.plt_validation_curve(x_train,y_train,'metric', metric_range)
weights_range = ['uniform','distance']
knnc.plt_validation_curve(x_train,y_train,'weights', weights_range)
knnc.evaluate(x_train, y_train, x_test, y_test, n_neighbors=11, metric='manhattan')