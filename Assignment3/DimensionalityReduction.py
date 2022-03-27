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
from clustering import clustering
import mlrose_hiive as mh
from scipy import stats
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

class D_reduction:
    def __init__(self):
        self.cluster = clustering()
        self.state = 42
    def preprossingdata(self, filename, filepath, test_size=0.2, unused_col=[], encode_col=[]):
        df = pd.read_csv(filepath + filename)
        # print(df.head())
        df_copy = df.copy()
        # print(df_copy.head())
        col_names = df_copy.columns

        last = df_copy[[col_names[-1]]]
        print(last.nunique())
        print(last.value_counts())
        plt.figure()
        last.value_counts().plot(kind='pie', title='Categories vs Counts', y='Counts')
        plt.tight_layout()
        plt.savefig(filename + 'Data Distribution.png')
        if encode_col != None:
            for i in encode_col:
                labelset = list(set(df_copy.iloc[:, i]))
                le = LabelEncoder()
                le.fit(labelset)
                df_copy.iloc[:, i] = le.transform(df_copy.iloc[:, i])
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


    def comp_2_visualizer(self, x , y, algo_name, fig_name):
        if algo_name == 'PCA':
            model = decomposition.PCA(n_components=2, random_state=self.state)
        elif algo_name == 'ICA':
            model = decomposition.FastICA(n_components=2, random_state=self.state)
        elif algo_name == 'RP':
            model = random_projection.GaussianRandomProjection(n_components=2, random_state=self.state)
        elif algo_name == 'FA':
            model = decomposition.FactorAnalysis(n_components=2, random_state=self.state)

        model.fit(x)
        x_2 = model.transform(x)
        x_2_df = pd.DataFrame(x_2, columns =['Principal Component 1', 'Principal Component 2'])
        targets = y_train.value_counts().index.tolist()

        plt.figure()
        for target in targets:
            indexes = np.where(y == target )
            plt.scatter(x_2_df.loc[indexes, 'Principal Component 1'], x_2_df.loc[indexes, 'Principal Component 2'],  s=30)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(['{}'.format(i) for i in targets])
        plt.title('{} Dataset: {} 2-Components Visualization'.format(fig_name.split('_')[0],algo_name))
        plt.tight_layout()
        plt.savefig('{}_{}_visualize.png'.format(fig_name,algo_name))

    def pca_components_plot(self, x , n_range, fig_name):

        pca = decomposition.PCA().fit(x)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        # print(pca.explained_variance_ratio_)
        plt.figure()
        fig, ax = plt.subplots()
        plt.axhline(y=0.95, linestyle= '--', color='r')
        ax.plot(cum_var, color='green', marker='o')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance', color='green')
        ax2 = ax.twinx()
        ax2.plot(pca.explained_variance_ratio_,color='blue', marker='o', linestyle='--')
        ax2.set_ylabel('Explained Variance', color='blue')
        plt.grid()
        plt.title('{} Dataset: PCA Dimensionality Reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_PCA.png'.format(fig_name))

    def ica_components_plot(self, x , n_range, fig_name):
        kur= []
        for n in n_range:
            ica = decomposition.FastICA(n_components=n, random_state=self.state).fit_transform(x)
            kurtosis = stats.kurtosis(ica)
            kur.append(np.mean(kurtosis))
        plt.figure()
        plt.subplots()
        plt.plot(n_range,kur)
        plt.xlabel('Number of Components')
        plt.ylabel('Kurtosis')
        plt.grid()
        plt.title('{} Dataset: ICA Dimensionality Reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_ICA.png'.format(fig_name))


    def random_projection_plot(self, x , n_range, fig_name):
        err_mtr = []
        seeds = [2,12,22,32,42]
        for seed in seeds:
            reconstruction_err_arr = []
            for n in n_range:
                rp = random_projection.GaussianRandomProjection(n_components=n, random_state=seed)
                x_transform = rp.fit_transform(x)

                reconstruction_err = np.mean((x-(x_transform.dot(rp.components_) + np.mean(x,axis=0)))**2)
                reconstruction_err_arr.append(reconstruction_err)
            err_mtr.append(reconstruction_err_arr)
        err_df = pd.DataFrame(err_mtr, columns=n_range, index=['seed={}'.format(i) for i in seeds]).T

        plt.figure()
        plt.subplots()
        # plt.plot(n_range,reconstruction_err_arr)
        err_df.plot()
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error')
        plt.grid()
        plt.title('{} Dataset: RP Dimensionality Reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_RP.png'.format(fig_name))

    def fa_components_plot(self, x , n_range, fig_name):
        loglike= []
        for n in n_range:
            fa = decomposition.FactorAnalysis(n_components=n, random_state=self.state).fit(x)
            loglike.append(np.mean(fa.loglike_))
        plt.figure()
        plt.subplots()
        plt.plot(n_range,loglike)
        plt.xlabel('Number of Components')
        plt.ylabel('LogLikehood')
        plt.grid()
        plt.title('{} Dataset: FA Dimensionality Reduction'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_FA.png'.format(fig_name))

    # def LDA_compnents_plot(self, x, y, n_range, fig_name):
    #
    #     exp_var_arr= []
    #     cum_var_arr= []
    #     for n in n_range:
    #         print(n)
    #         lda= LinearDiscriminantAnalysis(n_components=n)
    #         x_trans = lda.fit_transform(x, y)
    #         print(lda.get_params())
    #         exp_var_arr.append(lda.explained_variance_ratio_)
    #         cum_var_arr = np.cumsum(exp_var_arr)
    #
    #     plt.figure()
    #     fig, ax = plt.subplots()
    #     ax.plot(cum_var_arr, color='green', marker='o')
    #     ax.set_xlabel('Number of Components')
    #     ax.set_ylabel('Cumulative Explained Variance', color='green')
    #     ax2 = ax.twinx()
    #     ax2.plot(exp_var_arr,color='blue', marker='o', linestyle='--')
    #     ax2.set_ylabel('Explained Variance', color='blue')
    #     plt.grid()
    #     plt.title('LCA Dimensionality Reduction')
    #     plt.tight_layout()
    #     plt.savefig('{}_LCA.png'.format(fig_name))

    def pca_reduced_cluster(self, x, cluster, fig_name):
        n_range = range(2,11,2)
        k_range = range(2,19)
        sse_bic_mtx = []
        sil_mtx = []
        fittime_mtx = []
        for n in n_range:
            # pca = decomposition.PCA(n_components=0.95)
            pca = decomposition.PCA(n_components=n, random_state=self.state)
            pca_model = pca.fit(x)
            x_reduced = pca.transform(x)
            if cluster =='kmeans':
                sse, sil_coe, fit_time = self.cluster.reduced_kmeans_cluster(k_range, x_reduced)
                sse_bic_mtx.append(sse)
                label = 'SSE'
            elif cluster == 'em':
                bic, sil_coe, fit_time = self.cluster.reduced_em_cluster(k_range, x_reduced)
                sse_bic_mtx.append(bic)
                label = 'BIC'

            sil_mtx.append(sil_coe)
            fittime_mtx.append(fit_time)
        col_names = ['n_compnent = {}'.format(i) for i in n_range]
        sse_bic_df = pd.DataFrame(sse_bic_mtx,columns=['{}'.format(i) for i in k_range]).T
        sil_df =pd.DataFrame(sil_mtx,columns=['{}'.format(i) for i in k_range]).T
        fittime_df =pd.DataFrame(fittime_mtx,columns=['{}'.format(i) for i in k_range]).T

        sse_bic_df.columns = col_names
        sil_df.columns = col_names
        fittime_df.columns = col_names

        plt.figure(figsize=(18,5))
        ax1 = plt.subplot(1,3,1)
        sse_bic_df.plot(ax = ax1)
        plt.xlabel('Number of Clusters')
        plt.ylabel(label)
        plt.grid()

        ax2 = plt.subplot(1,3,2)
        sil_df.plot(ax=ax2)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Coefficient')
        plt.grid()

        ax3 = plt.subplot(1,3,3)
        fittime_df.plot(ax=ax3)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Fit Time')
        plt.grid()
        plt.title('{} Dataset: Clustering of PCA Dimensionality Reduced'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_PCA_Reduced_{}_Cluster.png'.format(fig_name,cluster))

    def reduced_cluster(self, x, reduce_algo, n_range, cluster, fig_name):
        k_range = range(2,19)
        sse_bic_mtx = []
        sil_mtx = []
        fittime_mtx = []
        for n in n_range:
            if reduce_algo == 'PCA':
                model = decomposition.PCA(n_components=n, random_state=self.state).fit(x)
            elif reduce_algo == 'ICA':
                model = decomposition.FastICA(n_components=n, random_state=self.state).fit(x)
            elif reduce_algo == 'RP':
                model = random_projection.GaussianRandomProjection(n_components=n, random_state=self.state).fit(x)
            elif reduce_algo == 'FA':
                model = decomposition.FactorAnalysis(n_components=n, random_state=self.state).fit(x)

            x_reduced = model.transform(x)

            if cluster =='kmeans':
                sse, sil_coe, fit_time = self.cluster.reduced_kmeans_cluster(k_range, x_reduced)
                sse_bic_mtx.append(sse)
                label = 'SSE'

            elif cluster == 'em':
                bic, sil_coe, fit_time = self.cluster.reduced_em_cluster(k_range, x_reduced)
                sse_bic_mtx.append(bic)
                label = 'BIC'

            sil_mtx.append(sil_coe)
            fittime_mtx.append(fit_time)
        col_names = ['n_compnent = {}'.format(i) for i in n_range]
        sse_bic_df = pd.DataFrame(sse_bic_mtx,columns=['{}'.format(i) for i in k_range]).T
        sil_df =pd.DataFrame(sil_mtx,columns=['{}'.format(i) for i in k_range]).T
        fittime_df =pd.DataFrame(fittime_mtx,columns=['{}'.format(i) for i in k_range]).T

        sse_bic_df.columns = col_names
        sil_df.columns = col_names
        fittime_df.columns = col_names

        knee_col = col_names[2:]
        for col in knee_col:
            kl = KneeLocator(k_range, sse_bic_df[col], curve='convex', direction='decreasing')
            print('{} {}: {} elbow: {}'.format(reduce_algo, col, cluster, kl.elbow))
            print('{} {}: {} max sil: {}'.format(reduce_algo, col, cluster, sil_df[col].idxmax()))

        plt.figure(figsize=(18,5))

        ax1 = plt.subplot(1,3,1)
        sse_bic_df.plot(ax = ax1)
        plt.xlabel('Number of Clusters')
        plt.ylabel(label)
        plt.grid()

        ax2 = plt.subplot(1,3,2)
        sil_df.plot(ax=ax2)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Coefficient')
        plt.grid()
        plt.title('{} Dataset: {} Clustering of {} Dimensionality Reduced Dataset'.format(fig_name.split('_')[0],cluster.upper(),reduce_algo))


        ax3 = plt.subplot(1,3,3)
        fittime_df.plot(ax=ax3)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Fit Time')
        plt.grid()

        plt.tight_layout()
        plt.savefig('{}_{}_Reduced_{}_Cluster.png'.format(fig_name,reduce_algo, cluster))


    def dim_reduction_cluster_analysis(self, x_train, fig_name):

        n_range = range(1,x_train.shape[1])

        # # Part1 : Run clustering Algorithms
        # self.cluster.kmeans_cluster(x_train, fig_name +'_Part1')
        # self.cluster.em_cluster(x_train, fig_name +'_Part1')
        #
        # # Part2: Run Dimensionality reduction Algorithms
        # self.pca_components_plot(x_train, n_range, fig_name +'_Part2')
        # self.ica_components_plot(x_train, n_range, fig_name +'_Part2')
        # self.random_projection_plot(x_train, n_range, fig_name +'_Part2')
        # self.fa_components_plot(x_train,n_range, fig_name +'_Part2')


        # Part3: Reproduce clustering with dimensionality reduced dataset
        n_range = range(1, x_train.shape[1], 2)
        n_range2 = range(2, x_train.shape[1], 2)

        self.reduced_cluster(x_train, reduce_algo='PCA',n_range = n_range, cluster='kmeans', fig_name = fig_name +'_Part3')
        self.reduced_cluster(x_train, reduce_algo='PCA',n_range = n_range, cluster='em', fig_name = fig_name +'_Part3')
        self.reduced_cluster(x_train, reduce_algo='ICA', n_range=n_range2, cluster='kmeans', fig_name=fig_name + '_Part3')
        self.reduced_cluster(x_train, reduce_algo='ICA', n_range=n_range2, cluster='em', fig_name=fig_name + '_Part3')
        self.reduced_cluster(x_train, reduce_algo='RP',n_range = n_range, cluster='kmeans', fig_name = fig_name +'_Part3')
        self.reduced_cluster(x_train, reduce_algo='RP',n_range = n_range, cluster='em', fig_name = fig_name +'_Part3')
        self.reduced_cluster(x_train, reduce_algo='FA', n_range=n_range, cluster='kmeans', fig_name=fig_name + '_Part3')
        self.reduced_cluster(x_train, reduce_algo='FA', n_range=n_range, cluster='em', fig_name=fig_name + '_Part3')


    def evaluated(self, x_train, y_train, reduce_algo, n, k_kmeans, k_em,fig_name):
        cluster_algos = ['kmeans', 'em']
        if reduce_algo == 'PCA':
            model = decomposition.PCA(n_components=n, random_state=self.state).fit(x_train)
        elif reduce_algo == 'ICA':
            model = decomposition.FastICA(n_components=n, random_state=self.state).fit(x_train)
        elif reduce_algo == 'RP':
            model = random_projection.GaussianRandomProjection(n_components=n, random_state=self.state).fit(x_train)
        elif reduce_algo == 'FA':
            model = decomposition.FactorAnalysis(n_components=n, random_state=self.state).fit(x_train)

        x_reduced = model.transform(x_train)
        homo_arr =[]
        for cluster in cluster_algos:
            if cluster == 'kmeans':
                for k in k_kmeans:
                    kmeans = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300, random_state=42)
                    kmeans.fit(x_reduced)
                    homo = metrics.homogeneity_score(y_train,kmeans.labels_)
                    homo_arr.append(homo)
            elif cluster == 'em':
                for k in k_em:
                    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state=42)
                    gmm.fit(x_reduced)
                    labels = gmm.fit_predict(x_reduced)
                    homo = metrics.homogeneity_score(y_train, labels)
                    homo_arr.append(homo)

        columns1= ['{} - k={}'.format(cluster_algos[0],k) for k in k_kmeans]
        columns2= ['{} - k={}'.format(cluster_algos[1],k) for k in k_em]
        # print(homo_arr)
        print(columns1+columns2)
        homo_df = pd.DataFrame(homo_arr, index=columns1+columns2, columns=['Homogeneity'])

        plt.figure()
        homo_df.plot(kind='bar')
        plt.xlabel('Clustering Method and Number')
        plt.xticks(rotation=45)
        plt.ylabel('Homogeneity Score')
        plt.title(('{} Dataset: Clustering Evaluation After {}'.format(fig_name,reduce_algo)))
        plt.tight_layout()
        plt.savefig('{}_{}_clustering evaluation'.format(fig_name,reduce_algo))




if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(2)
    test_size = 0.2
    filepath = './'

    pp = D_reduction()

    filename = 'Shill Bidding Dataset.csv'
    unused_col = [0, 2]
    x_train, x_test, y_train, y_test = pp.preprossingdata(filename, filepath, test_size=test_size,unused_col=unused_col)
    fig_name = filename.split(' ')[0]
    n_range = range(1, x_train.shape[1])
    # print()
    # pp.comp_2_visualizer(x_train, y_train, 'PCA', fig_name)
    # pp.comp_2_visualizer(x_train, y_train, 'FA', fig_name)
    # pp.dim_reduction_cluster_analysis(x_train, fig_name)

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
    # pp.dim_reduction_nn_analysis(x_train, x_test, y_train, y_test, fig_name+'_Part4')
    pp.dim_reduction_cluster_nn_analysis(x_train, x_test, y_train, y_test, 8, fig_name+'_Part5')

    # plt.show()
    t2 = time.time()
    print('Run Time: {}'.format(t2-t1))