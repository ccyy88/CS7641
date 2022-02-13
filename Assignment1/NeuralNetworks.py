import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, learning_curve,validation_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import metrics


"""hyperparameters:
(['activation', 'alpha', 'batch_size', 'beta_1', 'beta_2', 'early_stopping', 'epsilon', 'hidden_layer_sizes', 'learning_rate', 'learning_rate_init', 'max_fun', 'max_iter', 'momentum', 'n_iter_no_change', 'nesterovs_momentum', 'power_t', 'random_state', 'shuffle', 'solver', 'tol', 'validation_fraction', 'verbose', 'warm_start'])"""

class NeuralNetworkClf():
    def __init__(self,verbose = False):
        self.verbose = verbose

    # Create Neural Network classifier
    def classifier(self, x_train, y_train):
        clf = MLPClassifier()
        # clf = clf.fit(x_train, y_train)
        return clf


    def evaluate(self, x_train, y_train, x_test, y_test, hidden_layer_sizes, max_iter):
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
        with open(str(y_test.name)+'_test_results.txt', 'a') as file:
            file.writelines('\n')
            file.writelines(f"Neural Network Learner Metrics Report: \n")
            file.writelines(report)
            file.writelines('\n')

    # plot iterative learning curve for iterative algorithms
    def plt_learning_curve(self, x, y, cv=5, n_jobs=4,train_sizes = np.linspace(0.1,0.9,10)):
        d_clfs = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
        depth_range = range(10,500,30)
        for epoch in depth_range:
            clf = MLPClassifier( random_state = 0, warm_start=True, early_stopping=False, max_iter=epoch)
            clf.fit(x_train, y_train)
            d_clfs.append(clf)

        training_score = [clf.score(x_train,y_train) for clf in d_clfs]
        testing_score = [clf.score(x_test,y_test) for clf in d_clfs]
        fig, ax = plt.subplots()
        ax.plot(depth_range, training_score, marker = 'o', label = 'Train')
        ax.plot(depth_range, testing_score, marker = 'o', label = 'Validation')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.grid()
        plt.title('Neural Networks Learning Curve')
        fig.tight_layout()
        plt.savefig(str(y.name) + '_Neural Networks _learning_curve.png')


    #
    # # plot learning curve - regular way
    # def plt_learning_curve(self, x, y, cv=5, n_jobs=4,train_sizes = np.linspace(0.1,0.9,10)):
    #     estimator =MLPClassifier(random_state = 0, warm_start=True, early_stopping=False)
    #     train_sizes_percentage = train_sizes
    #     train_sizes,train_scores,validation_scores = learning_curve(estimator,x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
    #
    #     train_score_mean = np.mean(train_scores, axis=1)
    #     train_score_std = np.std(train_scores, axis=1)
    #     val_score_mean = np.mean(validation_scores, axis=1)
    #     val_score_std = np.std(validation_scores, axis=1)
    #     if self.verbose == True:
    #         print('Std: \n',train_score_std,'\n', val_score_std)
    #     # plot learning curve
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel('Training size percentage')
    #     ax.set_ylabel('Accuracy')
    #     ax.plot(train_sizes_percentage, train_score_mean, 'o-',label='Training')
    #     ax.plot(train_sizes_percentage, val_score_mean, 'o-',label='Cross-Validation')
    #     ax.legend()
    #     plt.grid()
    #     plt.title('Neural Networks Learning Curve')
    #     fig.tight_layout()
    #     plt.savefig(str(y.name)+'Neural Networks _learning_curve.png')
    #


    def plt_validation_curve(self, x, y, param_name, param_range):
        if param_name == 'batch_size':
            solver = 'sgd'
        else:
            solver = 'adam'
        estimator =MLPClassifier(random_state = 0, warm_start=True, solver=solver)#, max_iter=200, early_stopping=False)
        train_scores,validation_scores = validation_curve(estimator,x, y, param_name = param_name, param_range=param_range, scoring='accuracy', n_jobs=-1, cv=5)

        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        # if self.verbose == True:
        #     print('Std: \n',train_score_std,'\n', val_score_std)
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy')
        xrange = range(0,len(param_range))
        plt.xticks(xrange, param_range)
        ax.plot(xrange, train_score_mean, 'o-',label='Training')
        ax.plot(xrange, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('Neural Networks Validation Curve - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name)+'_Neural Networks_validation_curve_{}.png'.format(param_name))

        fig, ax = plt.subplots()
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy Stdev of Cross-Validation')
        xrange = range(0, len(param_range))
        plt.xticks(xrange, param_range)
        ax.plot(xrange, train_score_std, 'o-', label='Training')
        ax.plot(xrange, val_score_std, 'o-', label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('Neural Networks Validation Curve Std - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name) + '_Neural Networks_validation_curve_std_{}.png'.format(param_name))


    def gridsearch(self, x_tr, y_tr,cv=3):
        clf = MLPClassifier(max_iter=300)
        print(clf.get_params().keys())
        param_dict = {'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)],
                      'solver':['lbfgs', 'sgd', 'adam'], 'learning_rate':['constant', 'adaptive','logistic'], 'activation':['tanh','relu']}
        grid = GridSearchCV(clf,param_grid=param_dict, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(x_tr,y_tr)
        print(grid.best_params_)
        print(grid.best_estimator_)
        print(grid.best_score_)


# MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 20, 20), learning_rate='adaptive', max_iter=4)


# df = pd.read_csv('diabetes.csv')
# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# df.columns = col_names
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# print(df.head())
# print(df.describe().transpose())
#
# data_x = df[feature_cols]
# data_y = df.label
#
# # Normalize data
# data_x = data_x/data_x.max()
# print(data_x.describe().transpose())
#
# # Split data
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=40)
#
# # Building Neural Network Model
# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
# mlp.fit(x_train, y_train)
#
# y_train_pred = mlp.predict(x_train)
# y_test_pred = mlp.predict(x_test)
#
# print(confusion_matrix(y_train, y_train_pred))
# print(classification_report(y_train, y_train_pred))
#
# print(confusion_matrix(y_test, y_test_pred))
# print(classification_report(y_test, y_test_pred))
