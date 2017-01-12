""" Methods for doing logistic regression."""

import numpy as np
import math
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    # print 'log_predict data.shape:', data.shape
    y = sigmoid(np.dot(data, weights))

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    N, M = targets.shape
    opp_targets = np.subtract(np.ones((N, 1), dtype=np.float32), targets)
    opp_y = np.subtract(np.ones((N, 1), dtype=np.float32), y)
    ce = -(np.dot(targets.T, np.log(y)) + np.dot(opp_targets.T, np.log(opp_y)))
    ce = ce[0][0]

    y_labels = (y >= 0.5).astype(np.int)
    correct = 0
    for i in range(len(targets)):
        if y_labels[i] == targets[i]:
            correct += 1
    frac_correct = float(correct)/len(targets)
    return ce, frac_correct


def logistic(weights, data, targets, hyper_parameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyper_parameters: The hyper_parameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyper_parameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyper_parameters)
    else:
        # TODO: compute f and df without regularization
        N, M = data.shape
        # print 'logistic data shape', data.shape
        opp_targets = np.subtract(np.ones((N, 1), dtype=np.float32), targets)
        opp_y = np.subtract(np.ones((N, 1), dtype=np.float32), y)
        f = -(np.dot(targets.T, np.log(y)) + np.dot(opp_targets.T, np.log(opp_y)))
        f = f[0][0]

        d = np.subtract(y, targets)
        df = np.dot(data.T, d)

    return f, df, y


def logistic_pen(weights, data, targets, hyper_parameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyper_parameters: The hyper_parameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)
    N, M = data.shape
    opp_targets = np.subtract(np.ones((N, 1), dtype=np.float32), targets)
    opp_y = np.subtract(np.ones((N, 1), dtype=np.float32), y)

    alpha = hyper_parameters['weight_decay']
    penalty = -0.5 * alpha * np.dot(weights.T, weights) + 0.5 * np.log(2 * math.pi * alpha**-1)
    f = -(np.dot(targets.T, np.log(y)) + np.dot(opp_targets.T, np.log(opp_y))) - penalty
    f = f[0][0]

    d = np.subtract(y, targets)
    df = np.dot(data.T, d) + alpha * weights
    
    return f, df
