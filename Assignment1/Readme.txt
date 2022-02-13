Dataset 1 download URL:
https://www.kaggle.com/aishu2218/shill-bidding-dataset

Dataset 2 download URL:
https://www.kaggle.com/ankurbajaj9/obesity-levels

Environment: python > 3.6, scikit-learn 1.0.2

The project contains the following files:
    dataset1.py
        -- This file wraps up all the required experiments for Dataset1 and generate figures. The hyperparameters are tuned specifically for Dataset1.
    dataset2.py
        -- This file wraps up all the required experiments for Dataset2 and generate figures. The hyperparameters are tuned specifically for Dataset2.
    preprocessing.py
        -- This file contains the functions for data preprocessing. It's called in dataset1.py/dataset2.py
    DecisionTree.py
        -- This file has a class which defines the functions and methods for Decision Tree Classifier. It's called in dataset1.py/dataset2.py
    NeuralNetwork.py
        -- This file has a class which defines the functions and methods for Neural Networks Classifier. It's called in dataset1.py/dataset2.py
    Boosting.py
        -- This file has a class which defines the functions and methods for AdaBoosting Classifier.  It's called in dataset1.py/dataset2.py
    SVM.py
        -- This file has a class which defines the functions and methods for SVM Classifier.  It's called in dataset1.py/dataset2.py
    KNN.py
        -- This file has a class which defines the functions and methods for KNN Classifier. It's called in dataset1.py/dataset2.py

How to run the code:
    1. download all the code files into the same foler in your local drive
    2. download the datasets into a sub-folder called 'dataset'
    3. run dataset1.py to generate all the outputs for Dataset1
    4. run dataset2.py to generate all the outputs for Dataset2
