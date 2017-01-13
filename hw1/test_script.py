import numpy as np
import math
import operator
from pprint import pprint
import matplotlib.pyplot as plt

from l2_distance import l2_distance
from plot_digits import *
from utils import *
from run_knn import run_knn
from logistic import *


def accuracy_test(k_set, train_data, train_targets, test_data):
    success = []
    for j in k_set:
        valid_labels = run_knn(j, train_data, train_targets, test_data)
        correct = 0
        for i in range(len(valid_targets)):
            if valid_labels[i] == valid_targets[i]:
                correct += 1
        success.append(float(correct) / len(valid_targets) * 100)
    return success

train_data, train_targets = load_train()
train_data_small, train_targets_small = load_train_small()
valid_data, valid_targets = load_valid()
test_data, test_targets = load_test()


k = [1, 3, 5, 7, 9]
k_star = [3, 5, 7]
valid_success = accuracy_test(k, train_data, train_targets, valid_data)
test_success = accuracy_test(k_star, train_data, train_targets, test_data)


fig = plt.figure(1)
fig.suptitle('Classification Rate vs. k')

width = 1/1.5

axx_1 = plt.subplot(211)
rects_1 = axx_1.bar(k, valid_success, width, color="blue")
ax_1 = fig.add_subplot(211)
ax_1.set_title('kNN on Validation Set')
ax_1.set_ylabel('Classification rate')
# ax_1.set_xlabel('k')

axx_2 = plt.subplot(212)
rects_2 = axx_2.bar(k_star, test_success, width, color="blue")
ax_2 = fig.add_subplot(212)
ax_2.set_title('kNN on Test Set (k* = 5)')
ax_2.set_ylabel('Classification rate')
ax_2.set_xlabel('k')


def autolabel(rects, axx_n):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axx_n.text(rect.get_x() + rect.get_width() / 2., 0.8 * height,
                   '%d' % int(height),ha='center', va='bottom')


autolabel(rects_1, axx_1)
autolabel(rects_2, axx_2)

fig = plt.gcf()
plt.show()

