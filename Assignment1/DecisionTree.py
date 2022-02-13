import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.model_selection import GridSearchCV, learning_curve,validation_curve
# from sklearn.externals. import StringIO


class DecisionTreeClf():
    def __init__(self,verbose = False):
        self.verbose = verbose

    # Create Decision Tree classifier
    def classifier(self, x_train, y_train):
        clf = DecisionTreeClassifier()
        # clf = clf.fit(x_train, y_train)
        return clf

    # Evaluate Decision Tree classifier
    def evaluate(self, x_train, y_train, x_test, y_test, max_depth, max_leaf_nodes):
        clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
        with open(str(y_test.name)+'_test_results.txt', 'w') as file:
            file.writelines('\n')
            file.writelines(f"DT Learner Metrics Report: \n")
            file.writelines(report)
            file.writelines('\n')

    # Visualizing Decision Tree
    def visualize(self, clf, savefile):
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                        feature_names=feature_cols, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(savefile + '.png')
        Image(graph.create_png())

    # Optimizing/Pruning the Decision Tree
    def prunedclf(self,  x_train, y_train, alpha=0.0, criterion = 'gini', max_depth=6):
        clf_pruned = DecisionTreeClassifier(ccp_alpha=alpha, criterion=criterion, max_depth=max_depth)
        clf_pruned = clf_pruned.fit(x_train, y_train)
        return clf_pruned

    # plot learning curve
    def plt_learning_curve(self, x, y, cv=5, n_jobs=-1,train_sizes = np.linspace(0.01,0.8,10)):
        estimator = DecisionTreeClassifier(ccp_alpha=0.0, criterion = 'gini', max_depth=5, random_state=0)
        # estimator = DecisionTreeClassifier()
        train_sizes_percentage = train_sizes
        train_sizes,train_scores,validation_scores,fit_times,score_times = learning_curve(estimator,x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=False, return_times=True)
        fit_times_mean = np.mean(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)


        if self.verbose == True:
            print('Std: \n',train_score_std.mean(),'\n', val_score_std.mean())
            print('Fit Times: \n', fit_times, )
            print('Score Times: \n',  score_times)

            print(train_score_mean)
            print(val_score_mean)
            print(train_sizes_percentage)
            print(train_sizes)


        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel('Training size')
        ax.set_ylabel('Accuracy')
        # ax.plot(train_sizes_percentage, train_score_mean, 'o-',label='Training')
        # ax.plot(train_sizes_percentage, val_score_mean, 'o-',label='Cross-Validation')
        ax.plot(train_sizes, train_score_mean, 'o-',label='Training')
        ax.plot(train_sizes, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('Decision Tree Learning Curve')
        fig.tight_layout()
        plt.savefig(str(y.name)+'_DT_learning_curve.png')

        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('Training size')
        ax1.set_ylabel('Time')
        ax1.plot(train_sizes, fit_times_mean, 'o-', label='Fit Time')
        ax1.plot(train_sizes, score_times_mean, 'o-', label='Score Time')
        ax1.legend()
        plt.grid()
        plt.title('Decision Tree Learning Curve - Time')
        fig1.tight_layout()
        plt.savefig(str(y.name) + '_DT_learning_curve_time.png')

    def plt_validation_curve(self, x, y, param_name, param_range):
        estimator = DecisionTreeClassifier()
        train_scores,validation_scores = validation_curve(estimator,x, y, param_name = param_name, param_range=param_range, scoring='accuracy', n_jobs=-1)

        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        if self.verbose == True:
            print('Std: \n',train_score_std.mean(),'\n', val_score_std.mean())
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy')
        ax.plot(param_range, train_score_mean, 'o-',label='Training')
        ax.plot(param_range, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('Decision Tree Validation Curve - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name)+'_DT_validation_curve_{}.png'.format(param_name))

    def gridsearch(self, x_tr, y_tr, cv=5):
        param_dict = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,15),'min_samples_leaf': range(1,5),'min_samples_split':range(1,10)}
        clf = DecisionTreeClassifier()
        print(clf.get_params().keys())
        grid = GridSearchCV(clf,param_grid=param_dict, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(x_tr,y_tr)
        print(grid.best_params_)
        print(grid.best_estimator_)
        print(grid.best_score_)


# clf = dtc.classifier(x_train,y_train)
# path = clf.cost_complexity_pruning_path(x_train,y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
# a_clfs = []
# for alpha in ccp_alphas:
#     clf = dtc.prunedclf(x_train, y_train, alpha = alpha, criterion = 'gini', max_depth= 10)
#     a_clfs.append(clf)
# # remove the trivial tree with one node
# a_clfs = a_clfs[:-1]
# ccp_alphas = ccp_alphas[:-1]
#
# training_score = [clf.score(x_train,y_train) for clf in a_clfs]
# testing_score = [clf.score(x_test,y_test) for clf in a_clfs]
# fig, ax = plt.subplots()
# ax.plot(ccp_alphas, training_score, marker = 'o', label = 'train')
# ax.plot(ccp_alphas, testing_score, marker = 'o', label = 'test')
# ax.legend()
# ax.set_xlabel('alpha')
# ax.set_ylabel('Accuracy')

# d_clfs = []
# depth_range = range(1,10)
# for max_dp in depth_range:
#     clf = dtc.prunedclf(x_train, y_train, alpha = 0.0, max_depth= max_dp)
#     d_clfs.append(clf)
#
# training_score = [clf.score(x_train,y_train) for clf in d_clfs]
# testing_score = [clf.score(x_test,y_test) for clf in d_clfs]
# fig, ax = plt.subplots()
# ax.plot(depth_range, training_score, marker = 'o', label = 'train')
# ax.plot(depth_range, testing_score, marker = 'o', label = 'test')
# ax.legend()
# ax.set_xlabel('Max_depth')
# ax.set_ylabel('Accuracy')
# plt.show()
