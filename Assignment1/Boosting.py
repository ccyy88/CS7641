from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, learning_curve,validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# scores = cross_val_score(clf, X, y, cv =5)
# print(scores.mean())
#

class BoostingClf():
    def __init__(self,verbose = False):
        self.verbose = verbose

    # Create Neural Network classifier
    def classifier(self, x_train, y_train):
        clf = AdaBoostClassifier()
        # clf = clf.fit(x_train, y_train)
        return clf


    def evaluate(self, x_train, y_train, x_test, y_test, n_estimators, learning_rate):
        base_learner = DecisionTreeClassifier(max_depth=3)
        clf = AdaBoostClassifier(base_estimator = base_learner, n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
        with open(str(y_test.name)+'_test_results.txt', 'a') as file:
            file.writelines('\n')
            file.writelines(f"AddBoosting Learner Metrics Report: \n")
            file.writelines(report)
            file.writelines('\n')

    # plot learning curve
    def plt_learning_curve(self, x, y, base_learner = DecisionTreeClassifier(max_depth=1), cv=5, n_jobs=4,train_sizes = np.linspace(0.1,0.9,9), ):
        estimator =AdaBoostClassifier(random_state = 0,base_estimator=base_learner)
        train_sizes_percentage = train_sizes
        train_sizes,train_scores,validation_scores = learning_curve(estimator,x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)

        train_score_mean = np.mean(train_scores, axis=1)
        train_score_std = np.std(train_scores, axis=1)
        val_score_mean = np.mean(validation_scores, axis=1)
        val_score_std = np.std(validation_scores, axis=1)
        if self.verbose == True:
            print('Std: \n',train_score_std,'\n', val_score_std)
        # plot learning curve
        fig, ax = plt.subplots()
        ax.set_xlabel('Training size percentage')
        ax.set_ylabel('Accuracy')
        ax.plot(train_sizes_percentage, train_score_mean, 'o-',label='Training')
        ax.plot(train_sizes_percentage, val_score_mean, 'o-',label='Cross-Validation')
        ax.legend()
        plt.grid()
        plt.title('Boosting Learning Curve')
        fig.tight_layout()
        plt.savefig(str(y.name)+'_Boosting_learning_curve.png')

    def plt_validation_curve(self, x, y, param_name, param_range, base_learner):

        estimator =AdaBoostClassifier(random_state = 0, base_estimator=base_learner, learning_rate=1)
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
        plt.title('Boosting Validation Curve - {}'.format(param_name))
        fig.tight_layout()
        plt.savefig(str(y.name)+'_Boosting_validation_curve_{}.png'.format(param_name))

    def gridsearch(self, x_tr, y_tr,cv=3):
        clf = AdaBoostClassifier()
        print(clf.get_params().keys())
        param_dict = { 'learning_rate':np.linspace(0.1,1,10), 'n_estimators':range(1,200,10)}
        grid = GridSearchCV(clf,param_grid=param_dict, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(x_tr,y_tr)
        print(grid.best_params_)
        print(grid.best_estimator_)
        print(grid.best_score_)
