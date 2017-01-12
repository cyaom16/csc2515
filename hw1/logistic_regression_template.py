import matplotlib.pyplot as plt

from check_grad import check_grad
from utils import *
from logistic import *


def run_logistic_regression(hyper_parameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_test()

    # N is number of examples; M is the number of features per example.
    n_train, M = train_inputs.shape
    n_valid, M = valid_inputs.shape
    train_design_col = np.ones((n_train, 1), dtype=np.float32)
    valid_design_col = np.ones((n_valid, 1), dtype=np.float32)
    # now size of train_inputs is N x (M+1), as a design matrix X, and w_0 will be corresponding to 1 in input data
    train_inputs = np.hstack((train_inputs, train_design_col))
    valid_inputs = np.hstack((valid_inputs, valid_design_col))
    # print 'train_inputs.shape:', train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    # test weights
    weights = np.random.randn(M+1, 1)*0.1
    weights /= np.max(weights)
    # print weights

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyper_parameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyper_parameters['num_iterations'], 5))
    for t in xrange(hyper_parameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyper_parameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyper_parameters['learning_rate'] * df / n_train
        # print 'updated weight shape', weights.shape

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / n_train, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / n_train, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
    return logging


def run_check_grad(hyper_parameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions+1)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyper_parameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyper_parameters
    hyper_parameters = {
                    'learning_rate': 0.005,
                    'weight_regularization': True,  # boolean, True for using Gaussian prior on weights
                    'num_iterations': 7000,
                    'weight_decay': 0.001  # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyper_parameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyper_parameters)
    logging /= num_runs

    # TODO generate plots
    iteration_array = range(1, hyper_parameters['num_iterations']+1);

    fig = plt.figure(1)
    fig.suptitle('Regularized Logistic Regression for mnist_train')
    plt.subplot(211)
    plt.plot(iteration_array, logging[:,1])
    ax_1 = fig.add_subplot(211)
    ax_1.set_title('Training Set')
    ax_1.set_ylabel('Cross Entropy')
    # ax_1.set_xlabel('k')

    plt.subplot(212)
    plt.plot(iteration_array, logging[:, 3])
    ax_2 = fig.add_subplot(212)
    ax_2.set_title('Test Set')
    ax_2.set_ylabel('Cross Entropy')
    ax_2.set_xlabel('# of Iterations')

    fig = plt.gcf()
    plt.show()

