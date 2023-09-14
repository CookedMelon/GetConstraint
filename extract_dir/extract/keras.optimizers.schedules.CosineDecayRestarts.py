@keras_export(
    "keras.optimizers.schedules.CosineDecayRestarts",
    "keras.experimental.CosineDecayRestarts",
)
class CosineDecayRestarts(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times initial learning rate as the new learning rate.
    Example usage:
    ```python
    first_decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_learning_rate,
          first_decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    def __init__(
        self,
        initial_learning_rate,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name=None,
    ):
        """Applies cosine decay with restarts to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
          t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period.
          m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the
            initial_learning_rate.
          name: String. Optional name of the operation. Defaults to 'SGDRDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)
            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps
            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul))
                        / tf.math.log(t_mul)
                    )
                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (
                        completed_fraction - sum_r
                    ) / t_mul**i_restart
                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart
                return i_restart, completed_fraction
            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True),
            )
            m_fac = m_mul**i_restart
            cosine_decayed = (
                0.5
                * m_fac
                * (
                    1.0
                    + tf.cos(
                        tf.constant(math.pi, dtype=dtype) * completed_fraction
                    )
                )
            )
            decayed = (1 - alpha) * cosine_decayed + alpha
            return tf.multiply(initial_learning_rate, decayed, name=name)
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name,
        }
# Note: this code is still used by V1 APIs.
class LinearCosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a linear cosine decay schedule.
    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417
    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a linear cosine decay
    function to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      linear_decay = (decay_steps - step) / decay_steps
      cosine_decay = 0.5 * (
          1 + cos(pi * 2 * num_periods * step / decay_steps))
      decayed = (alpha + linear_decay) * cosine_decay + beta
      return initial_learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.experimental.LinearCosineDecay(
        initial_learning_rate, decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        num_periods=0.5,
        alpha=0.0,
        beta=0.001,
        name=None,
    ):
        """Applies linear cosine decay to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          num_periods: Number of periods in the cosine part of the decay.
            See computation above.
          alpha: See computation above.
          beta: See computation above.
          name: String.  Optional name of the operation.  Defaults to
            'LinearCosineDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "LinearCosineDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            num_periods = tf.cast(self.num_periods, dtype)
            alpha = tf.cast(self.alpha, dtype)
            beta = tf.cast(self.beta, dtype)
            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            linear_decayed = (decay_steps - global_step_recomp) / decay_steps
            completed_fraction = global_step_recomp / decay_steps
            fraction = 2.0 * num_periods * completed_fraction
            cosine_decayed = 0.5 * (
                1.0 + tf.cos(tf.constant(math.pi, dtype=dtype) * fraction)
            )
            linear_cosine_decayed = (
                alpha + linear_decayed
            ) * cosine_decayed + beta
            return tf.multiply(
                initial_learning_rate, linear_cosine_decayed, name=name
            )
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "num_periods": self.num_periods,
            "alpha": self.alpha,
            "beta": self.beta,
            "name": self.name,
        }
# Note: this code is still used by V1 APIs.
class NoisyLinearCosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a noisy linear cosine decay schedule.
    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417
    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a noisy linear cosine decay
    function to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      linear_decay = (decay_steps - step) / decay_steps)
      cosine_decay = 0.5 * (
          1 + cos(pi * 2 * num_periods * step / decay_steps))
      decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
      return initial_learning_rate * decayed
    ```
    where eps_t is 0-centered gaussian noise with variance
    initial_variance / (1 + global_step) ** variance_decay
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.experimental.NoisyLinearCosineDecay(
        initial_learning_rate, decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        initial_variance=1.0,
        variance_decay=0.55,
        num_periods=0.5,
        alpha=0.0,
        beta=0.001,
        seed=None,
        name=None,
    ):
        """Applies noisy linear cosine decay to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          initial_variance: initial variance for the noise. See computation
            above.
          variance_decay: decay for the noise's variance. See computation above.
          num_periods: Number of periods in the cosine part of the decay.
            See computation above.
          alpha: See computation above.
          beta: See computation above.
          seed: Integer, optional random seed to enable deterministic behavior.
          name: String.  Optional name of the operation.  Defaults to
            'NoisyLinearCosineDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.initial_variance = initial_variance
        self.variance_decay = variance_decay
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.name = name
        self._random_generator = backend.RandomGenerator(seed)
    def __call__(self, step):
        with tf.name_scope(self.name or "NoisyLinearCosineDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            initial_variance = tf.cast(self.initial_variance, dtype)
            variance_decay = tf.cast(self.variance_decay, dtype)
            num_periods = tf.cast(self.num_periods, dtype)
            alpha = tf.cast(self.alpha, dtype)
            beta = tf.cast(self.beta, dtype)
            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            linear_decayed = (decay_steps - global_step_recomp) / decay_steps
            variance = initial_variance / (
                tf.pow(1.0 + global_step_recomp, variance_decay)
            )
            std = tf.sqrt(variance)
            noisy_linear_decayed = (
                linear_decayed
                + self._random_generator.random_normal(
                    linear_decayed.shape, stddev=std
                )
            )
            completed_fraction = global_step_recomp / decay_steps
            fraction = 2.0 * num_periods * completed_fraction
            cosine_decayed = 0.5 * (
                1.0 + tf.cos(tf.constant(math.pi, dtype=dtype) * fraction)
            )
            noisy_linear_cosine_decayed = (
                alpha + noisy_linear_decayed
            ) * cosine_decayed + beta
            return tf.multiply(
                initial_learning_rate, noisy_linear_cosine_decayed, name=name
            )
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "initial_variance": self.initial_variance,
            "variance_decay": self.variance_decay,
            "num_periods": self.num_periods,
            "alpha": self.alpha,
            "beta": self.beta,
            "seed": self.seed,
            "name": self.name,
        }
