'''
CS7641 Assignment3: Unsupervised Learning and Dimensionality Reduction
Author: S Yu
Date: Mar. 2022
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import silhouette_score
from sklearn import mixture
from kneed import KneeLocator

from sklearn import decomposition
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mlrose_hiive as mh
from scipy import stats
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from clustering import clustering
from DimensionalityReduction import D_reduction
from sklearn.metrics import pairwise
class ANN:
    def __init__(self):
        self.cluster = clustering()
        self.DimReduction = D_reduction()
        self.state = 42

    def neural_network(self,x_train, y_train, x_test, y_test):
        eva_lst = []
        for iter in [10,20,30,50,80]+list(range(100,2001,100)):
            # print(iter)
            model = mh.NeuralNetwork(max_iters=iter, hidden_nodes=[8, 4, 6], activation='relu', random_state = self.state )
            t1 = time.time()
            model.fit(x_train, y_train)
            t2 = time.time()
            y_pred = model.predict(x_test)
            t3 = time.time()
            fit_time = t2- t1
            pred_time = t3 -t2
            test_score = metrics.log_loss(y_test, y_pred)
            eva_lst.append([iter, model.loss, test_score, fit_time, pred_time])
        df = pd.DataFrame(eva_lst, columns=['Iterations', 'train loss', 'validation loss','train time', 'test time'])
        df.set_index('Iterations', inplace=True)
        return df

    # plot iterative learning curve for iterative algorithms
    def plt_learning_curve(self, x_train, x_test, y_train, y_test,n, fig_name):
        depth_range = range(10,1000,50)
        reduce_algos=['Original', 'PCA', 'ICA', 'RP', 'FA']

        training_arr = []
        testing_arr = []


        for algo in reduce_algos:
            if algo == 'Original':
                x_train_reduced = x_train
                x_test_reduced = x_test
            else:
                if algo == 'PCA':
                    model = decomposition.PCA(n_components=11, random_state=self.state).fit(x_train)
                elif algo == 'ICA':
                    model = decomposition.FastICA(n_components=10, random_state=self.state).fit(x_train)
                elif algo == 'RP':
                    model = random_projection.GaussianRandomProjection(n_components=9, random_state=self.state).fit(x_train)
                elif algo == 'FA':
                    model = decomposition.FactorAnalysis(n_components=11, random_state=self.state).fit(x_train)
                x_train_reduced = model.transform(x_train)
                x_test_reduced = model.transform(x_test)
            d_clfs = []

            for epoch in depth_range:
                clf = MLPClassifier( random_state = self.state,hidden_layer_sizes=[8, 4, 6], early_stopping=False, max_iter=epoch)
                clf.fit(x_train_reduced, y_train)
                d_clfs.append(clf)


            training_score = [clf.score(x_train_reduced,y_train) for clf in d_clfs]
            testing_score = [clf.score(x_test_reduced,y_test) for clf in d_clfs]

            training_arr.append(training_score)
            testing_arr.append(testing_score)


        training_df = pd.DataFrame(training_arr, index=[i +'_Train' for i in reduce_algos ], columns=depth_range).T
        testing_df = pd.DataFrame(testing_arr, index=[i +'_Validation' for i in reduce_algos], columns=depth_range).T

        plt.figure(figsize=(24,18))
        ax = training_df.plot()
        testing_df.plot(ax=ax)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize='x-small',loc='lower right')
        plt.grid()
        plt.title(fig_name + '_Neural Networks Learning & Validation Curve\n Dimensionality Reduction'.format(n))
        plt.tight_layout()
        plt.savefig(fig_name + '_Neural Networks _learning_curve.png')

        # plot iterative learning curve for iterative algorithms

    def plt_cluster_learning_curve(self, x_train, x_test, y_train, y_test, fig_name, cluster_alone):
        depth_range = range(10, 1000, 50)
        reduce_algos = ['Original', 'PCA', 'ICA', 'RP', 'FA']

        training_arr = []
        testing_arr = []
        wall_clock =[]
        for algo in reduce_algos:
            if algo == 'Original':
                x_train_reduced = x_train
                x_test_reduced = x_test
                n=x_train.shape[1]
            else:
                if algo == 'PCA':
                    model = decomposition.PCA(n_components=11, random_state=self.state).fit(x_train)
                    n=11
                elif algo == 'ICA':
                    model = decomposition.FastICA(n_components=10, random_state=self.state).fit(x_train)
                    n=10
                elif algo == 'RP':
                    model = random_projection.GaussianRandomProjection(n_components=9, random_state=self.state).fit( x_train)
                    n=9
                elif algo == 'FA':
                    model = decomposition.FactorAnalysis(n_components=11, random_state=self.state).fit(x_train)
                    n=11
                x_train_reduced = model.transform(x_train)
                x_test_reduced = model.transform(x_test)

            x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=['Component {}'.format(i) for i in range(n)])
            x_test_reduced_df = pd.DataFrame(x_test_reduced, columns=['Component {}'.format(i) for i in range(n)])
            if algo == 'Original':
                clustering = KMeans(init='random', n_clusters=7, n_init=10, max_iter=300, random_state=self.state)
            elif algo == 'PCA':
                clustering = KMeans(init='random', n_clusters=15, n_init=10, max_iter=300, random_state=self.state)
            elif algo == 'ICA':
                clustering = KMeans(init='random', n_clusters=8, n_init=10, max_iter=300, random_state=self.state)
            elif algo == 'RP':
                clustering = KMeans(init='random', n_clusters=6, n_init=10, max_iter=300, random_state=self.state)
            elif algo == 'FA':
                clustering = KMeans(init='random', n_clusters=18, n_init=10, max_iter=300, random_state=self.state)

            clustering.fit(x_train_reduced)
            labels_df = pd.DataFrame(clustering.labels_, columns=['Cluster'])

            if cluster_alone == False:
                x_train_labels_df = pd.concat([x_train_reduced_df, labels_df], axis=1)
            else:
                if algo == 'Original':
                    x_train_labels_df = x_train_reduced_df
                    x_test_labels_df = x_test_reduced_df
                else:
                    # x_train_labels_df = labels_df.copy()
                    distance_arr =[]
                    for x in x_train_reduced:
                        distance =[]
                        for centroid in clustering.cluster_centers_:
                            # print(x, centroid)
                            # dis = pairwise.manhattan_distances(x, centroid)
                            dis = np.sqrt( sum([(x[i] - centroid[i])**2 for i in range(len(x))]))
                            distance.append(dis)
                        distance_arr.append(distance)
                    x_train_labels_df = pd.DataFrame(distance_arr)
                    print(x_train_labels_df.shape)

            clustering.fit(x_test_reduced)
            tlabels_df = pd.DataFrame(clustering.labels_, columns=['Cluster'])

            if cluster_alone == False:
                x_test_labels_df = pd.concat([x_test_reduced_df, tlabels_df], axis=1)
            else:
                if algo == 'Original':
                    x_train_labels_df = x_train_reduced_df
                    x_test_labels_df = x_test_reduced_df
                else:
                    # x_test_labels_df = tlabels_df.copy()
                    distance_arr =[]
                    for x in x_test_reduced:
                        distance =[]
                        for centroid in clustering.cluster_centers_:
                            # dis = pairwise.euclidean_distances(x, centroid)
                            dis = np.sqrt( sum([(x[i] - centroid[i])**2 for i in range(len(x))]))

                            distance.append(dis)
                        distance_arr.append(distance)
                    x_test_labels_df = pd.DataFrame(distance_arr)


            if algo == 'Original':
                x_train_labels_df = x_train_reduced_df
                x_test_labels_df = x_test_reduced_df

            d_clfs = []
            t = time.time()
            for epoch in depth_range:
                clf = MLPClassifier(random_state=self.state, hidden_layer_sizes=[8, 4, 6], early_stopping=False,max_iter=epoch)
                clf.fit(x_train_labels_df, y_train)
                d_clfs.append(clf)

            accuracy, fittime = self.nn_accuracy(x_train_labels_df, y_train, x_test_labels_df, y_test)

            # wall_clock.append(time.time() - t)
            wall_clock.append(fittime)
            training_score = [clf.score(x_train_labels_df, y_train) for clf in d_clfs]
            testing_score = [clf.score(x_test_labels_df, y_test) for clf in d_clfs]

            training_arr.append(training_score)
            testing_arr.append(testing_score)

        training_df = pd.DataFrame(training_arr, index=[i + '_Train' for i in reduce_algos], columns=depth_range).T
        testing_df = pd.DataFrame(testing_arr, index=[i + '_Validation' for i in reduce_algos], columns=depth_range).T

        plt.figure(figsize=(24, 18))
        ax = training_df.plot()
        testing_df.plot(ax=ax)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize='x-small', loc='lower right')
        plt.grid()
        plt.title(fig_name + '_Neural Networks Learning & Validation Curve\n Dimenstionality Reduction and Clustering'.format(n))
        plt.tight_layout()
        plt.savefig(fig_name + '_NN_learning_curve with Dim_reduction and Clustering.png')

        print(wall_clock)
        wall_clock_df = pd.DataFrame(wall_clock, index=reduce_algos)
        plt.figure()
        wall_clock_df.plot(kind = 'bar')
        plt.ylabel('Wall Clock Time')
        plt.title('{}: Neural Networks Run Time \nAfter Dim_Reduction and Cluster'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig(fig_name + '_NN Run Time DR and Cluster.png')

    def nn_accuracy(self, x_train, y_train, x_test, y_test):

        # model = mh.NeuralNetwork(max_iters=1000, hidden_nodes=[8, 4, 6], activation='relu', random_state = self.state)
        model = MLPClassifier(hidden_layer_sizes=[8, 4, 6], max_iter=1000)
        t = time.time()
        model.fit(x_train, y_train)
        fittime = time.time() - t

        y_pred = model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        return accuracy, fittime

    def nn_plot(self, df_pca, df_ica, df_rp, df_fa, feature):
        fig, ax = plt.subplots()

        df_pca[feature].plot(ax=ax, label = 'PCA')
        df_ica[feature].plot(ax=ax, label = 'ICA')
        df_rp[feature].plot(ax=ax, label = 'RP')
        df_fa[[feature]].plot(ax=ax, label = 'FA')

        ax.set_title('Neural Network-{}'.format(feature))
        ax.set(xlabel='Iteration', ylabel=feature)
        ax.grid()

        ax.legend(['PCA', 'ICA', 'RP', 'FA'])
        fig.savefig('NN {}.png'.format(feature))

    # Part4: Run neural network learner with dimensionality reduced dataset
    def dim_reduction_nn_analysis(self, x_train, x_test, y_train, y_test, fig_name):
        reduce_algos=['PCA', 'ICA', 'RP', 'FA']
        n_comp_arr = []
        n_range = range(2,x_train.shape[1],2)
        wall_clock_arr = []
        for n in n_range:
            algo_acc_arr = []
            wall_clock = []

            for algo in reduce_algos:
                if algo == 'PCA':
                    model = decomposition.PCA(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'ICA':
                    model = decomposition.FastICA(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'RP':
                    model = random_projection.GaussianRandomProjection(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'FA':
                    model = decomposition.FactorAnalysis(n_components=n, random_state=self.state).fit(x_train)

                x_train_reduced = model.transform(x_train)
                x_test_reduced = model.transform(x_test)
                x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=['Component {}'.format(i) for i in range(n)])
                x_test_reduced_df = pd.DataFrame(x_test_reduced, columns=['Component {}'.format(i) for i in range(n)])


                accuracy, fittime= self.nn_accuracy(x_train_reduced_df, y_train, x_test_reduced_df, y_test)
                algo_acc_arr.append(accuracy)

                wall_clock.append(fittime)

            n_comp_arr.append(algo_acc_arr)
            wall_clock_arr.append(wall_clock)

        acc_df = pd.DataFrame(n_comp_arr, columns=reduce_algos,index=n_range)
        plt.figure()
        acc_df.plot()
        plt.xlabel('n_components')
        plt.ylabel('NN Accuracy')
        plt.grid()
        plt.title('{}: Neural Network & Dimensionality_reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_NN_Dim_Reduction.png'.format(fig_name))

        wall_clock_df = pd.DataFrame(wall_clock_arr, columns=reduce_algos,index=n_range)
        plt.figure()
        wall_clock_df.plot(kind = 'bar')
        plt.ylim([0,5])
        plt.xlabel('n_components')
        plt.ylabel('Wall Clock Time')
        plt.title('{}: Neural Networks Run Time After Dim_Reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig(fig_name + '_Neural Networks Run Time.png')



    # Part5: Apply clustering algorithm as new feature and rerun neural network
    def dim_reduction_cluster_nn_analysis(self, x_train, x_test, y_train, y_test, n, fig_name):
        reduce_algos = ['PCA', 'ICA', 'RP', 'FA']
        cluster_algos = ['kmeans', 'em']

        k_range = range(2,21)
        algo_acc_arr = []

        for k in k_range:
            cluster_acc = []

            for algo in reduce_algos:
                if algo == 'PCA':
                    model = decomposition.PCA(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'ICA':
                    model = decomposition.FastICA(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'RP':
                    model = random_projection.GaussianRandomProjection(n_components=n, random_state=self.state).fit(x_train)
                elif algo == 'FA':
                    model = decomposition.FactorAnalysis(n_components=n, random_state=self.state).fit(x_train)

                x_train_reduced = model.transform(x_train)
                x_test_reduced = model.transform(x_test)
                x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=['Component {}'.format(i) for i in range(n)])
                x_test_reduced_df = pd.DataFrame(x_test_reduced, columns=['Component {}'.format(i) for i in range(n)])
                for c_algo in cluster_algos:
                    if c_algo == 'kmeans':
                        clustering = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300, random_state=self.state)
                        clustering.fit(x_train_reduced)
                        labels = pd.DataFrame(clustering.labels_, columns=['Cluster'])
                        clustering.fit(x_test_reduced)
                        tlabels = pd.DataFrame(clustering.labels_, columns=['Cluster'])
                    elif c_algo == 'em':
                        clustering = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state=self.state)
                        labels = clustering.fit_predict(x_train_reduced)
                        tlabels = clustering.fit_predict(x_test_reduced)
                        labels = pd.DataFrame(labels, columns=['Cluster'])
                        tlabels = pd.DataFrame(tlabels, columns=['Cluster'])

                    x_train_reduced_df = pd.concat([x_train_reduced_df, labels], axis = 1)
                    x_test_reduced_df = pd.concat([x_test_reduced_df, tlabels], axis=1)

                    accuracy,fittime = self.nn_accuracy(x_train=x_train_reduced_df, y_train=y_train, x_test = x_test_reduced_df, y_test= y_test)
                    cluster_acc.append(accuracy)

            algo_acc_arr.append(cluster_acc)

        acc_df = pd.DataFrame(algo_acc_arr, columns=[r_algo+'-'+c_algo for r_algo in reduce_algos for c_algo in cluster_algos], index=k_range)
        # acc_df = pd.DataFrame(x_train)
        plt.figure(figsize=(24,18))
        acc_df.plot()
        plt.xlabel('Number of Clusters')
        plt.ylabel('NN Accuracy')
        plt.xlim([min(k_range), max(k_range)])
        plt.grid()
        # plt.text(3,6,''.format(n), fontsize=22, color = 'r')
        plt.title('{}: Neural Network & Dimensionality_Reduced+Cluster\n n_components={}'.format(fig_name.split('_')[0],n))
        plt.tight_layout()
        plt.savefig('{}_NN_Dim_Reduction_Cluster.png'.format(fig_name))



if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(2)
    test_size = 0.2
    filepath = './'

    pp = D_reduction()
    nn = ANN()
    filename = 'Shill Bidding Dataset.csv'
    unused_col = [0, 2]
    x_train, x_test, y_train, y_test = pp.preprossingdata(filename, filepath, test_size=test_size,unused_col=unused_col)
    fig_name = filename.split(' ')[0]
    # n_range = range(1, x_train.shape[1])
    # print()
    # pp.comp_2_visualizer(x_train, y_train, 'PCA', fig_name)
    # pp.comp_2_visualizer(x_train, y_train, 'FA', fig_name)
    # # pp.dim_reduction_cluster_analysis(x_train, fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='PCA', n=7, k_kmeans=[3,7], k_em=[2,4],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='ICA', n=6, k_kmeans=[7,18], k_em=[2,4],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='RP', n=7, k_kmeans=[2,3], k_em=[2,6],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='FA', n=5, k_kmeans=[6,13], k_em=[3,6],fig_name=fig_name)
    # pp.dim_reduction_nn_analysis(x_train, x_test, y_train, y_test, fig_name+'_Part4')
    # pp.dim_reduction_cluster_nn_analysis(x_train, x_test, y_train, y_test, 3, fig_name+'_Part5')


    filename = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    unused_col = []
    encode_col = [0, 4, 5, 8, 9, 11, 14, 15, 16]
    x_train, x_test, y_train, y_test = pp.preprossingdata(filename, filepath, test_size=test_size, unused_col=unused_col, encode_col=encode_col)
    fig_name = filename.split('_')[0]
    # pp.comp_2_visualizer(x_train, y_train, 'PCA', fig_name)
    # pp.comp_2_visualizer(x_train, y_train, 'FA', fig_name)
    # pp.dim_reduction_cluster_analysis(x_train, fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='PCA', n=11, k_kmeans=[7,15], k_em=[2,6],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='ICA', n=10, k_kmeans=[3,8], k_em=[2,3],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='RP', n=9, k_kmeans=[2,6], k_em=[2,3],fig_name=fig_name)
    # pp.evaluated(x_train, y_train, reduce_algo='FA', n=11, k_kmeans=[7,18], k_em=[8,14],fig_name=fig_name)
    # nn.dim_reduction_nn_analysis(x_train, x_test, y_train, y_test, fig_name+'_Part4')
    # nn.dim_reduction_cluster_nn_analysis(x_train, x_test, y_train, y_test, 10, fig_name+'_Part5')
    # nn.plt_learning_curve(x_train, x_test, y_train, y_test,11,fig_name)
    # nn.plt_cluster_learning_curve(x_train, x_test, y_train, y_test, fig_name, cluster_alone=True)
    nn.plt_cluster_learning_curve(x_train, x_test, y_train, y_test, fig_name, cluster_alone=False)

    t2 = time.time()
    print('Run Time: {}'.format(t2-t1))