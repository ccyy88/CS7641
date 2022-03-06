import pandas as pd
import matplotlib.pyplot as plt
from OptAlgorithm import OptAlgos
from NN_Algos import nn_opt
import numpy as np
import time


if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(2)
    # Part 1: Optimization Problems and Algorithms
    Opt = OptAlgos()
    Opt.hyperparametertuning()
    Opt.compare_algos()

    # Part 2: Neural Network Optimization
    nn = nn_opt()
    nn.dataset_test()
    t2 = time.time()
    print(t2-t1)