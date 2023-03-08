import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from training import TrainWrapper, get_fashion_mnist
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def plot(iteration, min_x, max_x, data_x, data_y, gpr, num_points=50, log_scale=True):
    """ Plot the mean, uncertainty, samples and acquisition function.

    :param iteration: current iteration to
    :param min_x: lower input bound
    :param max_x: upper input bound
    :param data_x: list of all previous input points
    :param data_y: list of all previous output points
    :param gpr: Gaussian process
    :param num_points: number of points to plot
    :param log_scale: whether the log-scale was used for the input space
    """

    x = np.linspace(min_x, max_x, num_points)
    gpr_mean, gpr_std = gpr.predict(np.reshape(x, [len(x), -1]), return_std=True)
    acq = expected_improvement(np.reshape(x, [len(x), -1]), np.min(np.array(data_y)), gpr)

    fig, ax = plt.subplots(nrows=2, sharex="all")
    fig.suptitle("Iteration " + str(iteration))

    if log_scale:
        x = np.exp(x)
        data_x = np.reshape(data_x, [len(data_x), -1])
        min_x = np.exp(min_x)
        max_x = np.exp(max_x)
        data_x = np.exp(np.array(data_x).astype("float"))
        [ax_i.set_xscale('log') for ax_i in ax]

    ax[0].plot(x, -gpr_mean)
    ax[0].fill_between(x, -gpr_mean - gpr_std, - gpr_mean + gpr_std, alpha=0.2)
    ax[0].scatter(data_x, -np.array( data_y))
    ax[0].set_ylabel("accuracy")
    ax[0].set_xlim([min_x, max_x])

    ax[1].plot(x, acq )
    ax[1].set_ylabel("expected improvement")
    ax[1].set_xlabel("learning rate")
    ax[1].set_xlim([min_x, max_x])

    plt.savefig('img/iteration_' + str(iteration) + '.png')
    plt.close()


def expected_improvement(x, current_min, gpr):
    """ Expected improvement acquisition function.

    :param x: candidate point
    :param current_min: min
    :param gpr: Gaussian process
    """
    x = np.reshape(x, [len(x), -1])
    mean, std = gpr.predict(x, return_std=True)
    gamma = (current_min - mean) / std
    return std * (gamma * norm.cdf(gamma) + norm.pdf(gamma))


def get_query(gpr, min_x, max_x,  data_y, n=50):
    """ Returns the minimum of the expected improvement acquisition function.

    :param gpr: Gaussian process
    :param min_x: lower input bound
    :param max_x: upper input bound
    :param data_y: list of all previous output samples
    :param n: number of random restarts of the optimization process
    :return: optimum of the acquisition function
    """
    current_min_x = None
    current_min_y = np.inf
    for i in range(n):
        x_O = np.random.uniform(min_x, max_x)
        res = minimize(lambda x: -expected_improvement(x, np.min(data_y), gpr), x_O, method='L-BFGS-B',
                        bounds=[(min_x, max_x), ])
        if res.fun < current_min_y:
            current_min_x = res.x
            current_min_y = res.fun
    return current_min_x


def bayesian_optimization(initial_x, min_x, max_x, num_iter, model_wrapper, log_scale=True):
    """ Runs the Bayesian optimization.

    :param initial_x: initial guess
    :param min_x: lower input bound
    :param max_x:  upper input bound
    :param num_iter: maximum number of function evaluations
    :param model_wrapper: interface to the training procedure
    :param log_scale: whether to use the log-scale for the input space
    """
    if log_scale:
        initial_x = np.log(initial_x)
        min_x = np.log(min_x)
        max_x = np.log(max_x)

    data_x = []
    data_y = []
    gpr = None
    for i in range(num_iter):

        if i == 0:
            query_x = np.array([initial_x,])
        else:
            query_x = get_query(gpr, min_x, max_x, data_y)
        print("\nBO iteration " + str(i), "query_lr:", np.exp(query_x) if log_scale else query_x)
        if log_scale:
            query_y = model_wrapper.eval(np.exp(query_x))
        else:
            query_y = model_wrapper.eval(query_x)

        data_x.append(query_x)
        data_y.append(query_y)
        gpr = GaussianProcessRegressor(random_state=0).fit(np.reshape(data_x, [len(data_x), -1]), data_y)
        if i > 0:
            plot(i, min_x, max_x, data_x, data_y, gpr, log_scale=log_scale)
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = get_fashion_mnist(32)
    wrapper = TrainWrapper(train_loader, test_loader, 10, epochs=1)
    bayesian_optimization(0.001, 0.0001, 1, 10, wrapper)

