import tensorflow as tf
from pysgmcmc.tensor_utils import pdist, squareform, median, vectorize, uninitialized_params
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
from pysgmcmc.samplers.base_classes import MCMCSampler


# XXX: Interface needs to change more: particles should be List[List[tensorflow.Variable]]
# where each inner list is one guess of a network.
# This would enable the bnn code to change such that SVGD becomes applicable
# to our BNN.


class SVGDSampler(MCMCSampler):
    """ Stein Variational Gradient Descent Sampler.

        See [1] for more details on stein variational gradient descent.\n

        [1] Q. Liu, D. Wang
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm. <https://arxiv.org/pdf/1608.04471>`_

    """
    def __init__(self, particles, cost_fun, tf_scope="default", batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.1),
                 alpha=0.9, fudge_factor=1e-6, session=tf.get_default_session(),
                 dtype=tf.float64, seed=None):
        """ Initialize the sampler parameters and set up a tensorflow.Graph
            for later queries.

        Parameters
        ----------
        particles : List[tensorflow.Variable]
            List of particles each representing a (different) guess of the
            target parameters of this sampler.

        cost_fun : callable
            Function that takes `params` of *one* particle as input and
            returns a 1-d `tensorflow.Tensor` that contains the cost-value.
            Frequently denoted with `U` in literature.

        batch_generator : iterable, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        alpha : float, optional
            TODO DOKU
            Defaults to `0.9`.

        fudge_factor : float, optional
            TODO DOKU
            Defaults to `1e-6`.

        session : tensorflow.Session, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        See Also
        ----------
        pysgmcmc.sampling.MCMCSampler:
            Base class for `SteinVariationalGradientDescentSampler` that
            specifies how actual sampling is performed (using iterator protocol,
            e.g. `next(sampler)`).

        """

        assert isinstance(alpha, (int, float))
        assert isinstance(fudge_factor, (int, float))
        # assert callable(cost_fun)

        # self.particles = tf.stack(particles)

        self.particles = particles

        def cost_fun_wrapper():
            return [cost_fun(particle) for particle in particles]
            # return tf.map_fn(lambda particle: cost_fun(particle), self.particles)

        cost_fun_wrapper.__name__ = "potential_energy"  # cost_fun.__name__

        self._init_basic(
            params=particles,
            cost_fun=cost_fun,  # cost_fun_wrapper,
            tf_scope=tf_scope,
            batch_generator=batch_generator,
            session=session, seed=seed, dtype=dtype,
            stepsize_schedule=stepsize_schedule
        )

        fudge_factor = tf.constant(
            fudge_factor, dtype=self.dtype, name="fudge_factor"
        )

        self.epsilon = tf.Variable(
            stepsize_schedule.initial_value, dtype=self.dtype, name="stepsize"
        )

        stack_vectorized_params = tf.stack([par[0] for par in particles])

        self.n_particles = tf.cast(
            # self.particles.shape[0], self.dtype
            stack_vectorized_params.shape[0], self.dtype
        )

        historical_grad = tf.get_variable(
            "historical_grad", stack_vectorized_params.shape, dtype=dtype,
            initializer=tf.zeros_initializer()
        )

        self.session.run(
            tf.variables_initializer([historical_grad, self.epsilon])
        )

        # lnpgrad = tf.reshape(tf.squeeze(tf.gradients(self.cost, self.particles)), [-1, 1])
        # lnpgrad = tf.reshape(tf.squeeze(tf.gradients(self.cost, particles)), [-1, 1])
        grads = []
        for i, cost in enumerate(cost_fun):
            grads.append(tf.concat([vectorize(gradient) for gradient in tf.gradients(cost, particles[i])], axis=0))
        lnpgrad = tf.reshape(tf.squeeze(grads), [-1, 1])

        kernel_matrix, kernel_gradients = self.svgd_kernel(stack_vectorized_params)

        grad_theta = tf.divide(
            tf.matmul(kernel_matrix, lnpgrad) + kernel_gradients,
            self.n_particles
        )

        historical_grad_t = tf.assign(
            historical_grad,
            alpha * historical_grad + (1. - alpha) * (grad_theta ** 2)
        )

        adj_grad = tf.divide(
            grad_theta,
            fudge_factor + tf.sqrt(historical_grad_t)
        )

        # for i, param in enumerate(self.params):
        #     self.theta_t[i] = tf.assign_sub(
        #         param,
        #         self.epsilon * adj_grad[i]
        #     )

        for i, particle in enumerate(self.particles):

            # self.theta_t[i] = [None] * len(particle)
            vectorized_Theta_t = tf.assign_sub(
                self.vectorized_params[i], self.epsilon * adj_grad[i]
            )
            start_idx = 0

            for j, param in enumerate(particle):
                flat_shape = tf.reduce_prod(param.shape)
                vectorized_param = vectorized_Theta_t[start_idx:start_idx+flat_shape]
                self.theta_t[i*len(particle) + j] = tf.assign(
                    param,
                    tf.reshape(vectorized_param, shape=param.shape),
                    name="theta_t_%d_%d" % (i, j)
                )
                start_idx += flat_shape

    def _init_basic(self, params, cost_fun, tf_scope="default", batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01),
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        # Sanitize inputs
        assert batch_generator is None or hasattr(batch_generator, "__next__")
        assert seed is None or isinstance(seed, int)

        # assert isinstance(session, (tf.Session, tf.InteractiveSession))
        assert isinstance(dtype, tf.DType)

        # assert callable(cost_fun)

        self.tf_scope = tf_scope

        self.dtype = dtype

        self.n_iterations = 0

        self.seed = seed

        assert hasattr(stepsize_schedule, "update")
        assert hasattr(stepsize_schedule, "__next__")
        assert hasattr(stepsize_schedule, "initial_value")

        self.stepsize_schedule = stepsize_schedule

        self.batch_generator = batch_generator
        self.session = session

        self.params = params

        # set up costs
        self.cost_fun = cost_fun
        self.cost = cost_fun # cost_fun(self.params)

        # compute vectorized clones of all parameters
        with tf.variable_scope(self.tf_scope, reuse=tf.AUTO_REUSE):
            # self.vectorized_params = [vectorize(param) for param in self.params]
            self.vectorized_params = []

            for i, param in enumerate(self.params):
                # self.vectorized_params.append(tf.concat(
                #     [tf.reshape(par.initialized_value(), (-1,)) for par in param], axis=0))
                self.vectorized_params.append(tf.get_variable(
                    initializer=tf.concat([tf.reshape(par.initialized_value(), (-1,)) for par in param], axis=0),
                    name="%s/particle_%s" % (self.tf_scope, i)
                ))

            # self.vectorized_params = tf.stack(self.vectorized_params)

            self.epsilon = tf.get_variable(
                initializer=self.stepsize_schedule.initial_value,
                dtype=self.dtype,
                name="epsilon",
                trainable=False
            )

        # Initialize uninitialized parameters before usage in any sampler.
        init = tf.variables_initializer(
            uninitialized_params(
                session=self.session,
                params=self.vectorized_params + [self.epsilon]
                # params=self.params + self.vectorized_params + [self.epsilon]
            )
        )
        self.session.run(init)

        # query this later to determine the next sample
        self.theta_t = [None] * len(params) * len(params[0])


    def svgd_kernel(self, particles):
        """ Calculate a kernel matrix with corresponding derivatives
            for the given `particles`.
            TODO: DOKU ON KERNEL TRICK

        Parameters
        ----------
        particles : TODO

        Returns
        ----------
        kernel_matrix : tf.Tensor
            TODO

        kernel_gradients : tf.Tensor
            TODO

        """
        euclidean_distances = pdist(particles)
        pairwise_distances = squareform(euclidean_distances) ** 2

        # kernel trick
        h = tf.sqrt(
            0.5 * median(pairwise_distances) / tf.log(self.n_particles + 1.)
        )

        kernel_matrix = tf.exp(-pairwise_distances / h ** 2 / 2)
        kernel_sum = tf.reduce_sum(kernel_matrix, axis=1)

        kernel_gradients = tf.add(
            -tf.matmul(kernel_matrix, particles),
            tf.multiply(particles, tf.expand_dims(kernel_sum, axis=1))
        )

        return kernel_matrix, kernel_gradients / (h ** 2)

    # XXX: Probably unnecessary. Changes should happen toplevel.
    # However using this to test *just* the svgd implementation and make
    # it conform to list of lists interface still seems reasonable.
    # Later: make BNN use multiple get_net calls to get variables
    # and extract appropriate groups from tf.trainable_variables
    # (use scope prefix)
    def _duplicate_variables(self, variables, duplicate_index):
        duplicate = []
        for var in variables:
            name = var.name.split(":")[0] + "_" + str(duplicate_index)
            dup_var = tf.get_variable(name, initializer=var.initializer._inputs[1])
            # session.run(tf.variables_initializer([dup_var]))
            duplicate.append(dup_var)
        return duplicate