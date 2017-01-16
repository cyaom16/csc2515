import numpy as np
import math
import operator
import matplotlib.pyplot as plt

from plot_digits import *
from utils import *
from logistic_regression_template import *


train_data, train_targets = load_train()
train_data_small, train_targets_small = load_train_small()
valid_data, valid_targets = load_valid()
test_data, test_targets = load_test()

data = train_data_small
targets = train_targets_small
N, M = data.shape

weights = np.random.randn(M+1, 1)
weights /= np.max(weights)

alpha = 0.1
penalty = -0.5 * alpha*np.dot(weights.T, weights) + 0.5 * np.log(2*math.pi*alpha**-1)
# run_check_grad(hyper_parameters)

print penalty
