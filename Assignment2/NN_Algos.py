import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, log_loss
import mlrose_hiive as mh
from NeuralNetworks import NeuralNetworkClf
from mlrose_hiive.algorithms.decay import GeomDecay
import time

class nn_opt():
    def __init__(self):
        self.seed = 5
    def nn_algos(self, algo_name,x_train, x_test, y_train, y_test, **kwargs):
        eva_lst = []
        for iter in [10,20,30,50,80]+list(range(100,2001,100)):
            # print(iter)
            model = mh.NeuralNetwork( **kwargs, algorithm=algo_name, max_iters=iter,hidden_nodes=[8, 4, 6], activation='relu', random_state = self.seed )
            t1 = time.time()
            model.fit(x_train, y_train)
            t2 = time.time()
            y_pred = model.predict(x_test)
            t3 = time.time()
            fit_time = t2- t1
            pred_time = t3 -t2
            test_score = log_loss(y_test, y_pred)
            eva_lst.append([iter, model.loss, test_score, fit_time, pred_time])
        df = pd.DataFrame(eva_lst, columns=['Iterations', 'train loss', 'validation loss','train time', 'test time'])
        df.set_index('Iterations', inplace=True)
        return df

    def nn_plot(self, df_rhc, df_ga, df_sa, df_gd, feature, plot_ga = True):
        fig, ax = plt.subplots()
        df_rhc[[feature]].plot(ax=ax, label = 'RHC')
        if plot_ga == True:
            df_ga[feature].plot(ax=ax, label = 'GA')
        df_sa[feature].plot(ax=ax, label = 'SA')
        df_gd[feature].plot(ax=ax, label = 'GD')
        ax.set_title('NN {}'.format(feature))
        ax.set(xlabel='Iteration', ylabel=feature)
        ax.grid()
        if plot_ga == True:
            ax.legend(['RHC', 'GA', 'SA', 'GD'])
            fig.savefig('NN {}.png'.format(feature))
        else:
            ax.legend(['RHC', 'SA', 'GD'])
            fig.savefig('NN {} wo GA.png'.format(feature))

    def preprossingdata(self, filename, filepath, test_size=0.2, unused_col=[], encode_col=[]):
        df = pd.read_csv(filepath + filename)
        # print(df.head())
        df_copy = df.copy()
        print(df_copy.head())
        col_names = df_copy.columns

        last = df_copy[[col_names[-1]]]
        print(last.nunique())
        print(last.value_counts())
        last.value_counts().plot(kind='pie', title='Categories vs Counts',y='Counts')
        plt.tight_layout()
        plt.savefig(filename+'Data Distribution.png')
        if encode_col !=None:
            for i in encode_col:
                labelset = list(set(df_copy.iloc[:,i]))
                le = LabelEncoder()
                le.fit(labelset)
                df_copy.iloc[:,i] = le.transform(df_copy.iloc[:,i])
                # print(x_data.iloc[:,0])

        # discard the unique ID columns
        feature_cols = col_names[0:-1]
        if unused_col != None:
            feature_cols = feature_cols.delete(unused_col)
        class_col = col_names[-1]
        x_data = df_copy[feature_cols]
        y_data = df_copy[class_col]


        print(x_data.columns)
        print(x_data.shape)
        print(y_data.shape)


        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

        print(x_train.shape)
        print(x_test.shape)

        # Normalize training and testing data separately
        stdscaler = StandardScaler()
        stdscaler.fit(x_train)
        x_train = stdscaler.transform(x_train)
        stdscaler.fit(x_test)
        x_test = stdscaler.transform(x_test)

        return x_train, x_test, y_train, y_test

    def dataset_test(self, filename = 'Shill Bidding Dataset.csv', unused_col=[0,2], test_size = 0.2):
        filepath= './'
        x_train, x_test, y_train, y_test = self.preprossingdata(filename, filepath, test_size=test_size, unused_col=unused_col)

        df_gd = self.nn_algos('gradient_descent', x_train, x_test, y_train, y_test)
        df_rhc = self.nn_algos('random_hill_climb', x_train, x_test, y_train, y_test, restarts=3)
        # print(df_rhc)
        df_ga  = self.nn_algos('genetic_alg',x_train, x_test, y_train, y_test, mutation_prob=0.3, pop_size = 200)
        # print(df_ga)
        df_sa  = self.nn_algos('simulated_annealing',x_train, x_test, y_train, y_test, schedule=GeomDecay(decay=0.99, min_temp=0.001))
        # print(df_sa)
        self.nn_plot(df_rhc,df_ga,df_sa,df_gd,  'train loss')
        self.nn_plot(df_rhc,df_ga,df_sa,df_gd,  'validation loss')
        self.nn_plot(df_rhc,df_ga,df_sa,df_gd,  'train time')
        self.nn_plot(df_rhc,df_ga,df_sa,df_gd,  'train time', plot_ga = False)
        self.nn_plot(df_rhc,df_ga,df_sa,df_gd,  'test time')


if __name__=='__main__':
    nn = nn_opt()
    nn.dataset_test()