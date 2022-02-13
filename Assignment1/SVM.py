from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve,validation_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import time


class SVMClf():
    def __init__(self,verbose = False):
        self.verbose = verbose

    # Create Neural Network classifier
    def classifier(self, x_train, y_train):
        clf = SVC()
        # clf = clf.fit(x_train, y_train)
        return clf

    def evaluate(self, x_train, y_train, x_test, y_test, kernel, C):
        clf = SVC(kernel=kernel, C = C, max_iter=70)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
        with open(str(y_test.name)+'_test_results.txt', 'a') as file:
            file.writelines('\n')
            file.writelines(f"SVM Learner Metrics Report: \n")
            file.writelines(report)
            file.writelines('\n')

    # plot iterative learning curve for iterative algorithms
    def plt_learning_curve(self, x, y, cv=5, n_jobs=4, train_sizes=np.linspace(0.1, 0.9, 10)):
        d_clfs = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
        iter_range = range(10, 200, 20)
        for iter in iter_range:
            clf =SVC(random_state = 0,max_iter=iter)
            clf.fit(x_train, y_train)
            d_clfs.append(clf)

        training_score = [clf.score(x_train, y_train) for clf in d_clfs]
        testing_score = [clf.score(x_test, y_test) for clf in d_clfs]
        fig, ax = plt.subplots()
        ax.plot(iter_range, training_score, marker='o', label='Train')
        ax.plot(iter_range, testing_score, marker='o', label='Validation')
        ax.legend()
        ax.set_xlabel('Iter')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.grid()
        plt.title('SVM Learning Curve')
        fig.tight_layout()
        plt.savefig(str(y.name) + '_SVM_learning_curve.png')



    #
    # # plot learning curve
    # def plt_learning_curve(self, x, y, cv=5, n_jobs=4,train_sizes = np.linspace(0.1,0.9,9)):
    #     estimator =SVC(random_state = 0)
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
    #     plt.title('SVM Learning Curve')
    #     fig.tight_layout()
    #     plt.savefig(str(y.name)+'_SVM_learning_curve.png')

    def plt_validation_curve(self, x, y, param_name, param_range):
        estimator =SVC(random_state = 0)
        train_scores,validation_scores = validation_curve(estimator,x, y, param_name = param_name, param_range=param_range, scoring='accuracy', n_jobs=-1)

        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        if self.verbose == True:
            print('Std: \n',train_score_std,'\n', val_score_std)
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy')
        # xrange = range(0,len(param_range))
        # plt.xticks(xrange, param_range)
        ax.plot(param_range, train_score_mean, 'o-',label='Training')
        ax.plot(param_range, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('SVM Validation Curve - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name)+'_SVM_validation_curve_{}.png'.format(param_name))

    def plt_kernel_fit_time(self, x, y):
        time_list = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
        kernel_type=['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernel_type:
            clf =SVC(kernel=kernel, random_state = 0)
            start_time = time.time()
            clf.fit(x_train, y_train)
            end_time = time.time()
            time_list.append(end_time-start_time)

        fig, ax = plt.subplots()
        ax.plot(kernel_type, time_list, marker='o', label='Train')
        # ax.plot(iter_range, testing_score, marker='o', label='Validation')
        ax.legend()
        ax.set_xlabel('Kernel Type')
        ax.set_ylabel('Training Time')
        ax.legend()
        plt.grid()
        plt.title('SVM Kernel-Time ')
        fig.tight_layout()
        plt.savefig(str(y.name) + '_SVM_time.png')

    def gridsearch(self, x_tr, y_tr,cv=3):
        clf = SVC()
        print(clf.get_params().keys())
        param_dict = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'max_iter':range(1,10)}
        grid = GridSearchCV(clf,param_grid=param_dict, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(x_tr,y_tr)
        print(grid.best_params_)
        print(grid.best_estimator_)
        print(grid.best_score_)

# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# svc_clf = SVC(kernel='linear', gamma='scale', shrinking=False)
# svc_clf.fit(X,y)
# print(svc_clf.coef_)
