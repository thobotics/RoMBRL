"""
This module contains adapter functions to obtain `pymc3.(Multi-)Trace` objects
from any of our samplers.
This allows us to use any diagnostics supported by `pymc3` to quantify
our samplers.
"""

import tensorflow as tf
import numpy as np
import pymc3
import logging


class PYSGMCMCTrace(object):
    """
    Adapter class to connect the worlds of pysgmcmc and pymc3.
    Represents a single chain/trace of samples obtained from a sgmcmc sampler.
    """
    def __init__(self, chain_id, samples, varnames=None):
        """ Set up a trace with given (unique) `chain_id` and sampled values
            `samples` that each represent a full sampler iteration
            sampling values for all variables with the given
            `varnames`.

        Parameters
        ----------
        chain_id : int
            A numeric id that uniquely identifies this chain/trace.

        samples : List[List]
            Single chain of samples extracted from
            a `pysgmcmc.MCMCSampler` instance.

        varnames : List[String] or NoneType, optional
            TODO: doku

        Examples
        ----------
        The following example shows a simple construction of a
        PYSGMCMCTrace from 2d dummy data:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0., name="x"), tf.Variable(0., name="y")]
        >>> names = [variable.name for variable in params]
        >>> dummy_samples = [[0., 0.], [0.2, -0.2], [0.3, -0.5], [0.1, 0.]]
        >>> trace = PYSGMCMCTrace(chain_id=0, samples=dummy_samples, varnames=names)
        >>> trace.n_vars, trace.varnames == names, len(trace.varnames) == trace.n_vars
        (2, True, True)

        If `varnames` is `None`, anonymous names resulting from enumerating all
        target parameters are used:

        >>> dummy_samples = [[0., 0.], [0.2, -0.2], [0.3, -0.5], [0.1, 0.]]
        >>> trace = PYSGMCMCTrace(chain_id=0, samples=dummy_samples, varnames=None)
        >>> trace.varnames
        ['0', '1']

        """

        self.chain = chain_id

        assert(hasattr(samples, "__len__")), "Samples needs to have a __len__ attribute."
        assert(len(samples) >= 1), "There needs to be at least one sample."

        self.samples = samples

        first_sample = self.samples[0]

        if isinstance(first_sample, (float, np.float32, np.float64)):
            # handle 1-d samples
            self.n_vars = 1
            self.samples = [
                [sample] for sample in self.samples
            ]
        else:
            self.n_vars = len(first_sample)

        assert(self.n_vars >= 1), "The first sample needs to have at least one variable."

        if varnames is None:
            # use anonymous variable names: enumerate
            logging.warning(
                "Variables in a trace were not named when instantiating "
                "a `pysgmcmc.diagnostics.sample_chain.PYSGMCMCTrace` "
                "from that trace. We will give them anonymous names "
                "by enumerating all target parameter dimensions."
            )
            self.varnames = [
                str(variable_index) for variable_index in range(self.n_vars)
            ]

        else:
            self.varnames = varnames

        assert len(self.varnames) == self.n_vars

    @classmethod
    def from_sampler(cls, chain_id, sampler, n_samples, keep_every=1, varnames=None):
        """
        Instantiate a trace with id `chain_id` by extracting `n_samples`
        from `sampler`.

        Parameters
        ----------
        chain_id : int
            A numeric id that uniquely identifies this chain/trace.

        sampler : pysgmcmc.sampling.MCMCSampler subclass
            A sampler used to generate samples for this trace.

        n_samples : int
            Number of samples to extract from `sampler` for this chain/trace.

        keep_every : int
            Keep every `keep_every`th sample in each chain.

        varnames : List[String] or NoneType, optional
            TODO: DOKU

        Returns
        ----------
        trace : PYSGMCMCTrace
            A wrapper around `n_samples` samples for variables with names
            `varnames` extracted from `sampler`.
            (Unique) chain id of this trace will be `chain_id`.

        Examples
        ----------
        Below we show how to use this classmethod to obtain a `PYSGMCMCTrace`.
        Furthermore, we demonstrate that we can apply burn-in steps prior
        to passing the sampler, which enables us to automatically thin/remove
        all burn-in samples prior to recording the actual trace.

        We start by defining our problem, as cost function we use the negative
        log likelihood of a mixture of gaussians (gmm1).

        >>> import tensorflow as tf
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.relativistic_sghmc import RelativisticSGHMCSampler
        >>> from pysgmcmc.diagnostics.objective_functions import gmm1_log_likelihood
        >>> gmm1_negative_log_likelihood = lambda *args, **kwargs: -gmm1_log_likelihood(*args, **kwargs)
        >>> session = tf.Session()
        >>> params = [tf.Variable(0., dtype=tf.float32, name="p")]

        Next, we set up our sampler and perform 100 steps of burn-in.
        We skip the samples obtained and do not record them as part of our
        `PYSGMCMCTrace`.

        >>> n_burn_in_steps = 100
        >>> sampler = RelativisticSGHMCSampler(params=params, cost_fun=gmm1_negative_log_likelihood, dtype=tf.float32, session=session)
        >>> session.run(tf.global_variables_initializer())
        >>> _ = islice(sampler, n_burn_in_steps)

        Finally, we extract a `PYSGMCMCTrace` from the (already burnt-in)
        sampler using `PYSGMCMCTrace.from_sampler`.

        >>> n_samples = 1000
        >>> varnames = ["p"]
        >>> chain_id = 1234  # unique id
        >>> trace = PYSGMCMCTrace.from_sampler(sampler=sampler, n_samples=n_samples, varnames=varnames, chain_id=chain_id)
        >>> session.close()
        >>> isinstance(trace, PYSGMCMCTrace), len(trace), trace.varnames
        (True, 1000, ['p'])

        """
        from itertools import islice
        samples = [
            sample for sample, _ in islice(sampler, n_samples)
        ]

        # ensure all sampler target parameters have a name
        # => tensorflow names variables automatically, so this assumption
        # is fair
        assert all(hasattr(param, "name") for param in sampler.params)

        # read variable names from sampler parameters
        if varnames is None:
            varnames = [
                param.name for param in sampler.params
            ]
        return PYSGMCMCTrace(chain_id, samples, varnames)

    def __getitem__(self, index):
        """
        Extract all samples for a target parameter at the given `index` in this trace.
        NOTE: This is equivalent to `trace.get_values(trace.varnames[index])`.

        Parameters
        ----------
        index : int
            Index of the target parameter for which we want to look up samples.

        Returns
        ----------
        trace_samples : list
            All samples for the parameter at the given `index` in this trace.

        Examples
        ----------

        >>> from numpy import allclose
        >>> varnames = ["x", "y"]
        >>> dummy_samples = [[0., 0.], [0.2, -0.2], [0.3, -0.5], [0.1, 0.]]
        >>> trace = PYSGMCMCTrace(chain_id=0, samples=dummy_samples, varnames=varnames)
        >>> allclose(trace[0], trace.get_values("x")), allclose(trace[1], trace.get_values("y"))
        (True, True)

        """
        assert isinstance(index, int)
        assert 0 <= index < len(self.varnames)

        return self.get_values(self.varnames[index])

    def _slice(self, slice_):
        """
        Slice this trace using slice indices in `slice_`.
        Slicing a trace effectively projects the trace onto the target parameters
        with the slice indices. All other target parameter samples are discarded
        and only the ones in the slice are kept.

        Parameters
        ----------
        slice_ : slice
            A slice use to index this trace.

        Returns
        -------
        sliced_trace : PYSGMCMCTrace
            A new trace that keeps only variable indices in the given `slice_`.

        """
        # XXX: Set chain_id to something unique, instead of self.chain
        return PYSGMCMCTrace(
            chain_id=self.chain,
            samples=self.samples[slice_],
            varnames=self.varnames[slice_]
        )

    def point(self, index):
        """TODO: Docstring for point.

        Parameters
        ----------
        index : TODO

        Returns
        ----------
        TODO

        """
        sample = self.samples[index]
        return {
            varname: sample[variable_index]
            for variable_index, varname in enumerate(self.varnames)
        }

    def __len__(self):
        """ Length of a trace/chain is the number of samples in it. """
        return len(self.samples)

    def get_values(self, varname, burn=0, thin=1):
        """
        Get all sampled values in this trace for variable with name `varname`.

        Parameters
        ----------
        varname : string
            Name of a given target parameter of the sampler.
            Usually, this corresponds to the `name` attribute of the
            `tensorflow.Variable` object for this target parameter.

        burn : int, optional
            Discard the first `burn` sampled values from this chain.
            Defaults to `0`.

        thin : int, optional
            Only return every `thin`th sampled value from this chain.
            Defaults to `1`.

        Returns
        ----------
        sampled_values : np.ndarray (N, D)
            All values for variable `varname` that were sampled
            in this chain.
            Formatted as (N, D) `numpy.ndarray` where
            `N` is the number of sampler steps in this chain and
            `D` is the dimensionality of variable `varname`.


        Examples
        ----------
        This method makes each variable in a trace accessible by its name:

        >>> import tensorflow as tf
        >>> graph = tf.Graph()
        >>> params = [tf.Variable(0., name="x"), tf.Variable(0., name="y")]
        >>> params[0].name, params[1].name
        ('x_1:0', 'y_1:0')

        These names can be used to index the trace and obtain all sampled
        values for the corresponding target parameter:

        >>> names = [variable.name for variable in params]
        >>> dummy_samples = [[0., 0.], [0.2, -0.2], [0.3, -0.5], [0.1, 0.]]
        >>> trace = PYSGMCMCTrace(chain_id=0, samples=dummy_samples, varnames=names)
        >>> trace.get_values(varname="x_1:0"), trace.get_values(varname="y_1:0")
        (array([ 0. ,  0.2,  0.3,  0.1]), array([ 0. , -0.2, -0.5,  0. ]))

        If a queried name does not correspond to any parameter in the trace,
        a `ValueError` is raised:

        >>> names = [variable.name for variable in params]
        >>> dummy_samples = [[0., 0.], [0.2, -0.2], [0.3, -0.5], [0.1, 0.]]
        >>> trace = PYSGMCMCTrace(chain_id=0, samples=dummy_samples, varnames=names)
        >>> trace.get_values(varname="FANTASYVARNAME")
        Traceback (most recent call last):
          ...
        ValueError: Queried `PYSGMCMCTrace` for values of parameter with name 'FANTASYVARNAME' but the trace does not contain any parameter of that name. Known variable names were: '['x_1:0', 'y_1:0']'

        """

        if varname not in self.varnames:
            raise ValueError(
                "Queried `PYSGMCMCTrace` for values of parameter with "
                "name '{name}' but the trace does not contain any "
                "parameter of that name. "
                "Known variable names were: '{varnames}'"
                .format(name=varname, varnames=self.varnames)
            )

        var_index = self.varnames.index(varname)

        return np.asarray(
            [sample[var_index] for sample in self.samples[burn::thin]]
        )


def pymc3_multitrace(get_sampler, n_chains=2, samples_per_chain=100,
                     keep_every=10,
                     parameter_names=None):
    """
    Extract chains from `sampler` and return them as `pymc3.MultiTrace` object.

    Parameters
    ----------
    get_sampler : callable
        A callable that takes a `tensorflow.Session` object as input
        and returns a (possibly already burnt-in) instance of a
        `pysgmcmc.sampling.MCMCSampler` subclass.

    parameter_names : List[String] or NoneType, optional
        List of names for each target parameter of the sampler.
        Defaults to `None`, which attempts to look the parameter names up
        from the target parameters of the sampler returned by `get_sampler`.

    Returns
    ----------
    multitrace : pymc3.backends.base.MultiTrace
        TODO: DOKU

    Examples
    ----------
    TODO ADD EXAMPLE

    """

    single_traces = []

    for chain_id in range(n_chains):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            sampler = get_sampler(session=session)
            session.run(tf.global_variables_initializer())
            trace = PYSGMCMCTrace.from_sampler(
                chain_id=chain_id,
                sampler=sampler,
                n_samples=samples_per_chain,
                keep_every=keep_every,
                varnames=parameter_names
            )

            single_traces.append(trace)

    return pymc3.backends.base.MultiTrace(single_traces)
