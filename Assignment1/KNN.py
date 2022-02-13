from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics



class KNNClf():
    def __init__(self,verbose = False):
        self.verbose = verbose

    # Create Neural Network classifier
    def classifier(self, x_train, y_train):
        clf = KNeighborsClassifier()
        # clf = clf.fit(x_train, y_train)
        return clf

    def evaluate(self, x_train, y_train, x_test, y_test, n_neighbors, metric):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
        with open(str(y_test.name)+'_test_results.txt', 'a') as file:
            file.writelines('\n')
            file.writelines(f"KNN Learner Metrics Report: \n")
            file.writelines(report)
            file.writelines('\n')

    # plot learning curve
    def plt_learning_curve(self, x, y, cv=5, n_jobs=4,train_sizes = np.linspace(0.1,0.9,10)):
        estimator =KNeighborsClassifier(n_neighbors=5)
        train_sizes_percentage = train_sizes
        train_sizes,train_scores,validation_scores, fit_times, score_times = learning_curve(estimator,x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc_ovr',return_times=True)

        fit_times_mean = np.mean(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        if self.verbose == True:
            print('Std: \n',train_score_std,'\n', val_score_std)
            print(train_sizes_percentage)
            print(train_sizes)
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel('Training size percentage')
        ax.set_ylabel('AUC')
        ax.plot(train_sizes_percentage, train_score_mean, 'o-',label='Training')
        ax.plot(train_sizes_percentage, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('KNN Learning Curve')
        fig.tight_layout()
        plt.savefig(str(y.name)+'_KNN_learning_curve.png')

        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('Training size')
        ax1.set_ylabel('Time')
        ax1.plot(train_sizes, fit_times_mean, 'o-', label='Fit Time')
        ax1.plot(train_sizes, score_times_mean, 'o-', label='Score Time')
        ax1.legend()
        plt.grid()
        plt.title('KNN Learning Curve - Time')
        fig1.tight_layout()
        plt.savefig(str(y.name) + '_KNN_learning_curve_time.png')


    def plt_validation_curve(self, x, y, param_name, param_range):
        estimator =KNeighborsClassifier()
        train_scores,validation_scores = validation_curve(estimator,x, y, param_name = param_name, param_range=param_range, scoring='roc_auc_ovr', n_jobs=-1)

        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        if self.verbose == True:
            print('Std: \n',train_score_std,'\n', val_score_std)
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel(param_name)
        ax.set_ylabel('AUC')
        # xrange = range(0,len(param_range))
        # plt.xticks(xrange, param_range)
        ax.plot(param_range, train_score_mean, 'o-',label='Training')
        ax.plot(param_range, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('KNN Validation Curve - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name)+'_KNN_validation_curve_{}.png'.format(param_name))

    def gridsearch(self, x_tr, y_tr,cv=3):
        clf = KNeighborsClassifier()
        print(clf.get_params().keys())
        param_dict = {'n_neighbors':range(1,20), 'leaf_size':range(1,10)}
        grid = GridSearchCV(clf,param_grid=param_dict, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(x_tr,y_tr)
        print(grid.best_params_)
        print(grid.best_estimator_)
        print(grid.best_score_)


# knn_clf = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
# knn_clf.fit(X, y)
# print(knn_clf.kneighbors_graph(X).toarray())
#'algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'radius'