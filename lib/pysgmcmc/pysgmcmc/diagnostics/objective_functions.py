from scipy.misc import logsumexp
import tensorflow as tf
import numpy as np
import functools


def to_negative_log_likelihood(log_likelihood_function):
    """ Decorator that converts a log likelihood into a negative log likelihood.
        Callable `log_likelihood_function` represents the log likelihood
        and a callable `negative_log_likelihood_function` with the same
        signature is returned.

    Parameters
    ----------
    log_likelihood_function : callable
        Callable that represents a log likelihood function.

    Returns
    -------
    negative_log_likelihood_function : callable
        Callable that returns negative log likelihood.

    Examples
    -------
    Wrapping a dummy log likelihood:

    >>> import numpy as np
    >>> log_likelihood = lambda a, b: np.log(a + b)
    >>> negative_log_likelihood = to_negative_log_likelihood(log_likelihood)
    >>> input_a, input_b = 4, 5
    >>> ll = log_likelihood(input_a, input_b)
    >>> nll = negative_log_likelihood(input_a, input_b)
    >>> np.allclose(-ll, nll)
    True

    The name attribute of the wrapped function remains unchanged:

    >>> log_likelihood.__name__ == negative_log_likelihood.__name__
    True

    """
    @functools.wraps(log_likelihood_function)
    def negative_log_likelihood(*args, **kwargs):
        return -log_likelihood_function(*args, **kwargs)
    return negative_log_likelihood


# XXX Give references: Relativistic Monte Carlo
def banana_log_likelihood(x):
    """
    Examples
    ----------

    >>> optimum, f_opt = (0, 10), 0.
    >>> np.allclose(banana_log_likelihood(optimum), f_opt)
    True

    """
    return -0.5 * (0.01 * x[0] ** 2 + (x[1] + 0.1 * x[0] ** 2 - 10) ** 2)


def gaussian_mixture_model_log_likelihood(x, mu=(-5, 0, 5), var=(1., 1., 1.),
                                          weights=(1. / 3., 1. / 3., 1. / 3.)):
    assert len(mu) == len(var) == len(weights)
    
    if hasattr(x, "__iter__"):
        # assert(len(x) == 1)
        x = x[0]

    if isinstance(x, tf.Variable) or isinstance(x, tf.Tensor):
        def normldf_tf(x, mu, var):
            pi = tf.constant(np.pi)
            return -0.5 * tf.log(2.0 * pi * var) - 0.5 * ((x - mu) ** 2) / var

        return tf.reduce_logsumexp(
            [tf.log(weights[i]) + normldf_tf(x, mu[i], var[i]) for i in range(len(mu))]
        )
    else:
        def normldf(x, mu, var):
            return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

        return logsumexp([
            np.log(weights[i]) + normldf(x, mu[i], var[i])
            for i in range(len(mu))
        ])


# def gaussian_mixture_model_log_likelihood(x, mu=(-5, 0, 5), var=(1., 1., 1.),
#                                           weights=(1. / 3., 1. / 3., 1. / 3.)):
#     assert len(mu) == len(var) == len(weights)

#     if hasattr(x, "__iter__"):
#         assert(len(x) == 1)
#         x = x[0]

#     if isinstance(x, tf.Variable):
#         def normldf_tf(x, mu, var):
#             pi = tf.constant(np.pi)
#             return -0.5 * tf.log(2.0 * pi * var) - 0.5 * ((x - mu) ** 2) / var

#         return tf.reduce_logsumexp(
#             [tf.log(weights[i]) + normldf_tf(x, mu[i], var[i]) for i in range(len(mu))]
#         )
#     else:
#         def normldf(x, mu, var):
#             return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

#         return logsumexp([
#             np.log(weights[i]) + normldf(x, mu[i], var[i])
#             for i in range(len(mu))
#         ])


# XXX: Give references for gmm functions (relativistic monte carlo paper)
def gmm1_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x)


def gmm2_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x, var=[1. / 0.5, 0.5, 1. / 0.5])


def gmm3_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x, var=[1. / 0.3, 0.3, 1. / 0.3])


def sinc(x):
    return np.sinc(x * 10 - 5).sum(axis=1)


# ## HPOLIB SYNTHETIC FUNCTIONS ## #

def bohachevski(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima, f_opt = [[0., 0.]], 0.0
    >>> np.allclose([bohachevski(optimum) for optimum in optima], f_opt)
    True

    """

    y = 0.7 + x[0] ** 2 + 2.0 * x[1] ** 2
    y -= 0.3 * np.cos(3.0 * np.pi * x[0])
    y -= 0.4 * np.cos(4.0 * np.pi * x[1])
    return y


def branin(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]
    >>> f_opt = 0.39788735773
    >>> np.allclose([branin(optimum) for optimum in optima], f_opt)
    True

    """
    y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
    return y


def camelback(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.0898, -0.7126], [-0.0898, 0.7126]]
    >>> f_opt = -1.03162842
    >>> np.allclose([camelback(optimum) for optimum in optima], f_opt)
    True

    """
    y = (4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * \
        (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * (x[1] ** 2)
    return y


def goldstein_price(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.0, -1.0]]
    >>> f_opt = 3.
    >>> np.allclose([goldstein_price(optimum) for optimum in optima], f_opt)
    True

    """
    y = (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))\
        * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    return y


def hartmann3(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.114614, 0.555649, 0.852547]]
    >>> f_opt = -3.8627795317627736
    >>> np.allclose([hartmann3(optimum) for optimum in optima], f_opt)
    True

    """
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1090, 8732, 5547],
                           [381, 5743, 8828]])
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(3):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)
    return -external_sum


def hartmann6(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]
    >>> f_opt = -3.322368011391339
    >>> np.allclose([hartmann6(optimum) for optimum in optima], f_opt)
    True

    """
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)
    return -external_sum


def levy(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[1.0]]
    >>> f_opt = 0.0
    >>> np.allclose([levy(optimum) for optimum in optima], f_opt)
    True

    """
    z = 1 + ((x[0] - 1.) / 4.)
    s = np.power((np.sin(np.pi * z)), 2)
    y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))
    return y


def rosenbrock(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[1, 1]]
    >>> f_opt = 0.0
    >>> np.allclose([rosenbrock(optimum) for optimum in optima], f_opt)
    True

    """
    y = 0
    d = 2
    for i in range(d - 1):
        y += 100 * (x[i + 1] - x[i] ** 2) ** 2
        y += (x[i] - 1) ** 2

    return y


def sin_one(x):
    """
    One dimensional sin function introduced in the paper:
        K. Kawaguchi, L. P. Kaelbling, and T. Lozano-Perez.
        Bayesian Optimization with Exponential Convergence.
        In Advances in Neural Information Processing (NIPS), 2015

    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.6330131633013163]]
    >>> f_opt = 0.042926342433644127
    >>> np.allclose([sin_one(optimum) for optimum in optima], f_opt)
    True

    """
    y = 0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5
    return y


def sin_two(x):
    """
    Two dimensional sin function introduced in the paper:
        K. Kawaguchi, L. P. Kaelbling, and T. Lozano-Perez.
        Bayesian Optimization with Exponential Convergence.
        In Advances in Neural Information Processing (NIPS), 2015

    Examples
    -------

    >>> import numpy as np
    >>> optima = [[0.6330131633013163, 0.6330131633013163]]
    >>> f_opt = 0.042926342433644127 ** 2
    >>> np.allclose([sin_two(optimum) for optimum in optima], f_opt)
    True

    """
    y = (0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5) * (0.5 * np.sin(13 * x[1]) * np.sin(27 * x[1]) + 0.5)
    return y
