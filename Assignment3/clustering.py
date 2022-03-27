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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn import mixture
from kneed import KneeLocator


class clustering:

    def preprossingdata(self, filename, filepath, test_size=0.2, unused_col=[], encode_col=[]):
        df = pd.read_csv(filepath + filename)
        # print(df.head())
        df_copy = df.copy()
        print(df_copy.head())
        col_names = df_copy.columns

        last = df_copy[[col_names[-1]]]
        print(last.nunique())
        print(last.value_counts())
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


    def kmeans_cluster(self, x,fig_name):
        sse = []
        silhouette_coe=[]
        k_range = range(2,19)
        for k in k_range:
            kmeans = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300, random_state=42)
            kmeans.fit(x)
            sse.append(kmeans.inertia_)
            sil_score = silhouette_score(x, kmeans.labels_)
            silhouette_coe.append(sil_score)
        kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
        print('kmeans elbow: {}'.format(kl.elbow))
        fig, ax = plt.subplots()
        ax.plot(k_range, sse, color='green',marker = 'o')
        ax.set_ylabel('SSE',color = 'green')
        ax.set_xlabel('Number of Clusters')

        ax2=ax.twinx()
        ax2.plot(k_range,silhouette_coe, color='blue',marker='o')
        ax2.set_ylabel('Silhouette Coefficient',color='blue')
        plt.grid()
        plt.title('{} Dataset: Kmeans_Cluster'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_Kmeans_Cluster'.format(fig_name))

        return sse, silhouette_coe

    def reduced_kmeans_cluster(self, k_range, x):
        sse = []
        silhouette_coe=[]
        fit_time = []
        for k in k_range:
            t_start = time.time()
            kmeans = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300, random_state=42)
            kmeans.fit(x)
            t_end = time.time()
            t_fit = t_end - t_start
            sil_score = silhouette_score(x, kmeans.labels_)

            fit_time.append(t_fit)
            sse.append(kmeans.inertia_)
            silhouette_coe.append(sil_score)

        return sse, silhouette_coe, fit_time


    def em_cluster(self, x, fig_name):
        bic = []
        silhouette_coe=[]
        n_range = range(2,19)
        for n in n_range:
            gmm = mixture.GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(x)
            bic.append(gmm.bic(x))
            labels = gmm.fit_predict(x)
            sil_score = silhouette_score(x, labels)
            silhouette_coe.append(sil_score)
        kl = KneeLocator(n_range, bic, curve='convex', direction='decreasing')
        print('em elbow: {}'.format(kl.elbow))

        fig, ax =plt.subplots()
        ax.plot(n_range, bic, color='green', marker = 'o')
        ax.set_ylabel('BIC', color='green')
        ax.set_xlabel('Number of components')

        ax2=ax.twinx()
        ax2.plot(n_range,silhouette_coe, color='blue', marker = 'o')
        ax2.set_ylabel('Silhouette Coefficient', color='blue')
        plt.grid()
        plt.title('{} Dataset: EM_Cluster'.format(fig_name.split('_')[0]))
        plt.tight_layout()
        plt.savefig('{}_EM_Cluster'.format(fig_name))

        return bic, silhouette_coe

    def reduced_em_cluster(self, k_range, x):
        bic = []
        silhouette_coe = []
        fit_time = []
        for k in k_range:
            t_start = time.time()
            gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(x)
            t_end = time.time()
            labels = gmm.fit_predict(x)
            t_fit = t_end - t_start
            sil_score = silhouette_score(x, labels)

            fit_time.append(t_fit)
            bic.append(gmm.bic(x))
            silhouette_coe.append(sil_score)

        return bic, silhouette_coe, fit_time

    def dataset_test(self, filename='Shill Bidding Dataset.csv', unused_col=[0, 2], test_size=0.2):
        filepath = './'
        x_train, x_test, y_train, y_test = self.preprossingdata(filename, filepath, test_size=test_size,
                                                                unused_col=unused_col)

        self.kmeans_cluster(x_train)
        self.em_cluster(x_train)

if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(2)

    pp = clustering()
    pp.dataset_test()


    plt.show()
    t2 = time.time()
    print('Run Time (min): {}'.format((t2-t1)/60))