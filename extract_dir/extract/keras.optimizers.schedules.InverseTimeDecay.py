@keras_export("keras.optimizers.schedules.InverseTimeDecay")
class InverseTimeDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses an inverse time decay schedule.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies the inverse decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate / (1 + decay_rate * step / decay_step)
    ```
    or, if `staircase` is `True`, as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.
    Example: Fit a Keras model when decaying 1/t with a rate of 0.5:
    ```python
    ...
    initial_learning_rate = 0.1
    decay_steps = 1.0
    decay_rate = 0.5
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
      initial_learning_rate, decay_steps, decay_rate)
    model.compile(optimizer=tf.keras.optimizers.SGD(
                      learning_rate=learning_rate_fn),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels, epochs=5)
    ```
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        name=None,
    ):
        """Applies inverse time decay to the initial learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: How often to apply decay.
          decay_rate: A Python number.  The decay rate.
          staircase: Whether to apply decay in a discrete staircase, as opposed
            to continuous, fashion.
          name: String.  Optional name of the operation.  Defaults to
            'InverseTimeDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "InverseTimeDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)
            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.floor(p)
            const = tf.cast(tf.constant(1), dtype)
            denom = tf.add(const, tf.multiply(decay_rate, p))
            return tf.divide(initial_learning_rate, denom, name=name)
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }
