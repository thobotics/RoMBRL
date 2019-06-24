import numpy as np
import tensorflow as tf
from itertools import islice
import pytest
import unittest

from pysgmcmc.data_batches import generate_batches
from pysgmcmc.tests.utils import sinc, init_random_uniform

try:
    from hypothesis import given
    from hypothesis.strategies import integers
except ImportError:
    HYPOTHESIS_INSTALLED = False
else:
    HYPOTHESIS_INSTALLED = True


def batch_generator(X=None, y=None, x_placeholder=None,
                    y_placeholder=None, seed=None, batch_size=10):
    return generate_batches(x=X, y=y,
                            x_placeholder=x_placeholder,
                            y_placeholder=y_placeholder,
                            seed=seed,
                            batch_size=batch_size)


def assert_batch_shapes(batch):
    x_placeholder, _ = tuple(batch.keys())
    X_batch, y_batch = tuple(batch.values())
    assert X_batch.shape[1] == x_placeholder.shape[1]
    assert y_batch.shape[0] == X_batch.shape[0] == y_batch.shape[0]


@pytest.mark.skipif(
    not HYPOTHESIS_INSTALLED, reason="Package 'hypothesis' not installed!"
)
class HypothesisTest(unittest.TestCase):
    """ Base class for property-based tests that use 'hypothesis' strategies. """
    @classmethod
    def random_nonint_input_type_strategy(cls):
        from hypothesis.strategies import (
            one_of, floats, complex_numbers, lists,
            sets, fractions, text
        )
        return one_of(
            floats(), complex_numbers(),
            lists(integers(), max_size=10), sets(integers(), max_size=10),
            fractions(), text()
        )

    def setup_data(self, seed=None, n_points=100):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # For train, step burn-in/step
        self.X = init_random_uniform(lower=np.zeros(1),
                                     upper=np.ones(1),
                                     n_points=n_points,
                                     rng=self.rng)
        self.y = sinc(self.X)

        # For self
        self.x_self = np.linspace(0, 1, 100)[:, None]

        self.y_self = sinc(self.x_self)

        n_inputs = self.X.shape[1]

        self.X_Placeholder = tf.placeholder(shape=(None, n_inputs), dtype=tf.float64,
                                            name="X_Minibatch")

        self.Y_Placeholder = tf.placeholder(dtype=tf.float64, name="Y_Minibatch")


@pytest.mark.skipif(
    not HYPOTHESIS_INSTALLED, reason="Package 'hypothesis' not installed!"
)
class HypothesisInvalidInputs(HypothesisTest):
    """ Property-based tests with invalid inputs"""
    @given(HypothesisTest.random_nonint_input_type_strategy())
    def test_invalid_input_types_batch_size(self, batch_size):
        with pytest.raises(AssertionError):
            next(batch_generator(batch_size=batch_size))

    @given(integers(max_value=0))
    def test_nonpositive_inputs_batch_size(self, batch_size):
        with pytest.raises(AssertionError):
            next(batch_generator(batch_size=batch_size))

    @given(HypothesisTest.random_nonint_input_type_strategy())
    def test_invalid_input_types_seed(self, seed):
        with pytest.raises(AssertionError):
            next(batch_generator(seed=seed))

    @given(integers(max_value=-1))
    def test_negative_inputs_seed(self, seed):
        with pytest.raises(AssertionError):
            next(batch_generator(seed=seed))


@pytest.mark.skipif(
    not HYPOTHESIS_INSTALLED, reason="Package 'hypothesis' not installed!"
)
class HypothesisTestSimpleBatches(HypothesisTest):
    """ Valid simple cases for batch extraction"""

    @given(
        integers(min_value=3, max_value=100),
        integers(min_value=0, max_value=2**32 - 1)
    )
    def test_simple_batch(self, n_points, seed):
        batch_size = np.random.randint(1, n_points)
        self.setup_data(n_points=n_points, seed=seed)
        generator = batch_generator(
            X=self.X, y=self.y,
            x_placeholder=self.X_Placeholder, y_placeholder=self.Y_Placeholder,
            seed=seed,
            batch_size=batch_size
        )
        assert_batch_shapes(next(generator))


@pytest.mark.skipif(
    not HYPOTHESIS_INSTALLED, reason="Package 'hypothesis' not installed!"
)
class HypothesisTestCornerCases(HypothesisTest):
    """
    Test that corner cases are handled properly, i.e. the batch size may be
    equal to or larger than the dataset size.
    """
    @given(
        integers(min_value=1, max_value=1000),
        integers(min_value=0, max_value=2**32 - 1)
    )
    def test_batchsize_equals_n_datapoints(self, n_points, seed):
        """ Extracting batches with batch_size == dataset size. """
        self.setup_data(n_points=n_points, seed=seed)
        generator = batch_generator(
            X=self.X, y=self.y,
            x_placeholder=self.X_Placeholder, y_placeholder=self.Y_Placeholder,
            seed=seed,
            batch_size=n_points
        )

        for _ in range(10):  # extract and check ten batches
            batch = next(generator)
            assert_batch_shapes(batch)
            assert np.allclose(batch[self.X_Placeholder], self.X, atol=1e-02)
            assert np.allclose(
                batch[self.Y_Placeholder], self.y.reshape(-1, 1), atol=1e-02
            )

    @given(
        integers(min_value=1, max_value=1000),
        integers(min_value=0, max_value=2**32 - 1)
    )
    def test_batch_size_larger_n_datapoints(self, n_points, seed):
        """ Extracting batches with batch_size > dataset size. """
        self.setup_data(n_points=n_points, seed=seed)

        batch_size = np.random.randint(n_points + 1, n_points * 100)
        generator = batch_generator(
            X=self.X, y=self.y,
            x_placeholder=self.X_Placeholder, y_placeholder=self.Y_Placeholder,
            seed=seed,
            batch_size=batch_size
        )
        for _ in range(10):  # extract and check ten batches
            batch = next(generator)
            assert_batch_shapes(batch)
            assert np.allclose(batch[self.X_Placeholder], self.X, atol=1e-02)
            assert np.allclose(
                batch[self.Y_Placeholder], self.y.reshape(-1, 1), atol=1e-02
            )


@pytest.mark.skipif(
    not HYPOTHESIS_INSTALLED, reason="Package 'hypothesis' not installed!"
)
class HypothesisSeededBatches(HypothesisTest):
    """ Test that fixing a seed makes batches reproducible. """

    @given(
        integers(min_value=1, max_value=1000),  # n_points
        integers(min_value=2, max_value=10),  # n_batches
        integers(min_value=1, max_value=1000),  # batch_size
        integers(min_value=2, max_value=10),  # n_generators
        integers(min_value=0, max_value=2**32 - 1)  # seed
    )
    def test_seeded_batches(self, n_points, n_batches, batch_size,
                            n_generators, seed):

        self.setup_data(seed=seed, n_points=n_points)
        generators = [
            batch_generator(
                X=self.X, y=self.y,
                x_placeholder=self.X_Placeholder, y_placeholder=self.Y_Placeholder,
                seed=seed,
                batch_size=batch_size
            )
            for _ in range(n_generators)
        ]

        for reference_batch in islice(generators[0], n_batches):
            reference = np.array(list(reference_batch.values()))
            for generator in generators[1:]:
                batch = np.array(list(next(generator).values()))
                assert np.allclose(reference, batch, atol=1e-02)
