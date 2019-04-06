# -*- coding: utf-8 -*-

"""
    data_batches.py - Custom generate_batches for SSM
    
    Created on  : February 03, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import numpy as np
import logging

__all__ = (
    "generate_batches",
    "generate_shuffled_batches",
)


def generate_batches(x, y, x_placeholder, y_placeholder, x_noise=None, x_noise_placeholder=None, n_points_placeholder=None, batch_size=20,
                     seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `x`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    batch_dict : dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `x`, `y`.

    Examples
    -------
    Simple batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    Batch extraction resizes batch size if dataset is too small:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((10, 3), (10,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((10, 3), (10, 1))

    In this case, the batches contain exactly all datapoints:

    >>> np.allclose(batch_dict[x_placeholder], x), np.allclose(batch_dict[y_placeholder].reshape(N,), y)
    (True, True)

    """

    # Sanitize inputs
    assert (isinstance(batch_size, int)), "generate_batches: batch size must be an integer."
    assert (batch_size > 0), "generate_batches: batch size must be greater than zero."

    assert (seed is None or isinstance(seed, int)), "generate_batches: seed must be an integer or `None`"

    assert seed is None or (0 <= seed <= 2 ** 32 - 1)

    assert (y.shape[0] == x.shape[0]), "Not exactly one label per datapoint!"

    n_examples = x.shape[0]

    if seed is None:
        seed = np.random.randint(1, 100000)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to %s", batch_size)

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds
        start = rng.randint(0, (n_examples - batch_size + 1))

        minibatch_x = x[start:start + batch_size]
        minibatch_y = y[start:start + batch_size]

        feed_dict = {
            x_placeholder: minibatch_x,
            y_placeholder: minibatch_y,
            n_points_placeholder: x.shape[0]
        }

        if x_noise is not None:
            minibatch_x_noise = x_noise[start:start + batch_size]
            feed_dict[x_noise_placeholder] = minibatch_x_noise

        yield feed_dict


def generate_weighted_batches(x, y, x_placeholder, y_placeholder, x_new=None, y_new=None,
                              weight_placeholder=None, weight=None,
                              continual_placeholder=None, n_points_placeholder=None, batch_size=20,
                     seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `x`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    batch_dict : dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `x`, `y`.

    Examples
    -------
    Simple batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    Batch extraction resizes batch size if dataset is too small:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((10, 3), (10,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((10, 3), (10, 1))

    In this case, the batches contain exactly all datapoints:

    >>> np.allclose(batch_dict[x_placeholder], x), np.allclose(batch_dict[y_placeholder].reshape(N,), y)
    (True, True)

    """

    # Sanitize inputs
    assert (isinstance(batch_size, int)), "generate_batches: batch size must be an integer."
    assert (batch_size > 0), "generate_batches: batch size must be greater than zero."

    assert (seed is None or isinstance(seed, int)), "generate_batches: seed must be an integer or `None`"

    assert seed is None or (0 <= seed <= 2 ** 32 - 1)

    assert (y.shape[0] == x.shape[0]), "Not exactly one label per datapoint!"

    n_examples = x.shape[0]

    if seed is None:
        seed = np.random.randint(1, 100000)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to %s", batch_size)

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds

        if x_new is None:
            start = rng.randint(0, (n_examples - batch_size + 1))

            minibatch_x = x[start:start + batch_size]
            minibatch_y = y[start:start + batch_size]
            minibatch_w = np.zeros((batch_size, weight.shape[1]))

            n_datapoints = x.shape[0]
            continual_train = False
        else:
            start0 = rng.randint(0, (len(x) - batch_size + 1))
            idx_old = slice(start0, start0 + batch_size)

            if len(x_new) < batch_size:
                idx_new = slice(0, len(x_new))
            else:
                start1 = rng.randint(0, (len(x_new) - batch_size + 1))
                idx_new = slice(start1, start1 + batch_size)

            minibatch_x = np.vstack([x[idx_old], x_new[idx_new]])
            minibatch_y = np.vstack([y[idx_old], y_new[idx_new]])
            minibatch_w = weight[idx_old] if weight is not None else weight

            n_datapoints = x.shape[0] + x_new.shape[0]
            continual_train = True

        feed_dict = {
            x_placeholder: minibatch_x,
            y_placeholder: minibatch_y,
            n_points_placeholder: n_datapoints,
            continual_placeholder: continual_train,
        }

        if weight_placeholder is not None:
            feed_dict[weight_placeholder] = minibatch_w

        yield feed_dict


def generate_cluster_batches(x, y, x_placeholder, y_placeholder, cluster, batch_size=20,
                     seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `x`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    batch_dict : dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `x`, `y`.

    Examples
    -------
    Simple batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    Batch extraction resizes batch size if dataset is too small:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((10, 3), (10,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((10, 3), (10, 1))

    In this case, the batches contain exactly all datapoints:

    >>> np.allclose(batch_dict[x_placeholder], x), np.allclose(batch_dict[y_placeholder].reshape(N,), y)
    (True, True)

    """

    # Sanitize inputs
    assert (isinstance(batch_size, int)), "generate_batches: batch size must be an integer."
    assert (batch_size > 0), "generate_batches: batch size must be greater than zero."

    assert (seed is None or isinstance(seed, int)), "generate_batches: seed must be an integer or `None`"

    assert seed is None or (0 <= seed <= 2 ** 32 - 1)

    assert (y.shape[0] == x.shape[0]), "Not exactly one label per datapoint!"

    n_examples = x.shape[0]

    if seed is None:
        seed = np.random.randint(1, 100000)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to %s", batch_size)

    y0 = y[np.where(cluster == 0)[0]]
    y1 = y[np.where(cluster == 1)[0]]

    x0 = x[np.where(cluster == 0)[0]]
    x1 = x[np.where(cluster == 1)[0]]

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds
        start = rng.randint(0, (n_examples - batch_size + 1))

        ystart0 = rng.randint(0, (len(y0) - batch_size + 1))
        ystart1 = rng.randint(0, (len(y1) - batch_size + 1))

        # minibatch_x = x[start:start + batch_size]
        minibatch_x = np.vstack([x0[ystart0:ystart0 + batch_size], x1[ystart1:ystart1 + batch_size]])
        minibatch_y = np.vstack([y0[ystart0:ystart0 + batch_size], y1[ystart1:ystart1 + batch_size]])

        # minibatch_cluster = cluster[start:start + batch_size]

        feed_dict = {
            x_placeholder: minibatch_x,
            y_placeholder: minibatch_y,
            # cluster_placeholder: minibatch_cluster,
        }

        yield feed_dict


def generate_z_batches(x, z, y, x_placeholder, z_placeholder, y_placeholder, x_noise=None, x_noise_placeholder=None, batch_size=20,
                     seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `x`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    batch_dict : dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `x`, `y`.

    Examples
    -------
    Simple batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    Batch extraction resizes batch size if dataset is too small:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((10, 3), (10,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((10, 3), (10, 1))

    In this case, the batches contain exactly all datapoints:

    >>> np.allclose(batch_dict[x_placeholder], x), np.allclose(batch_dict[y_placeholder].reshape(N,), y)
    (True, True)

    """

    # Sanitize inputs
    assert (isinstance(batch_size, int)), "generate_batches: batch size must be an integer."
    assert (batch_size > 0), "generate_batches: batch size must be greater than zero."

    assert (seed is None or isinstance(seed, int)), "generate_batches: seed must be an integer or `None`"

    assert seed is None or (0 <= seed <= 2 ** 32 - 1)

    assert (y.shape[0] == x.shape[0] == z.shape[0]), "Not exactly one label per datapoint!"

    n_examples = x.shape[0]

    if seed is None:
        seed = np.random.randint(1, 100000)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to %s", batch_size)

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds
        start = rng.randint(0, (n_examples - batch_size + 1))

        minibatch_x = x[start:start + batch_size]
        minibatch_z = z[start:start + batch_size]
        minibatch_y = y[start:start + batch_size]

        feed_dict = {
            x_placeholder: minibatch_x,
            z_placeholder: minibatch_z,
            y_placeholder: minibatch_y,
        }

        if x_noise is not None:
            minibatch_x_noise = x_noise[start:start + batch_size]
            feed_dict[x_noise_placeholder] = minibatch_x_noise

        yield feed_dict


def generate_shuffled_batches(x, y, x_placeholder, y_placeholder,
                              batch_size=20, seed=None):
    """ Infinite generator of shuffled random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `x`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation (and for shuffling!).
        Defaults to `None`.

    Yields
    -------
    batch_dict: dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `x`, `y`.

    Examples
    -------

    Simple shuffled batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> x = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> x.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_shuffled_batches(x, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    TODO: Demonstrate that shuffled batches are shuffled correctly, e.g.
    datapoint still matches corresponding label

    """

    # always use a seed in order to shuffle x and y in the same way
    if seed is None:
        seed = np.random.randint(1, 100000)

    rng_x, rng_y = np.random.RandomState(), np.random.RandomState()
    rng_x.seed(seed)
    rng_y.seed(seed)

    for batch in generate_batches(x, y, x_placeholder, y_placeholder, batch_size, seed):
        # shuffles x and y in the same way
        rng_x.shuffle(batch[x_placeholder])
        rng_y.shuffle(batch[y_placeholder])
        yield batch