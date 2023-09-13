@keras_export("keras.optimizers.schedules.LearningRateSchedule")
class LearningRateSchedule:
    """The learning rate schedule base class.
    You can use a learning rate schedule to modulate how the learning rate
    of your optimizer changes over time.
    Several built-in learning rate schedules are available, such as
    `tf.keras.optimizers.schedules.ExponentialDecay` or
    `tf.keras.optimizers.schedules.PiecewiseConstantDecay`:
    ```python
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    ```
    A `LearningRateSchedule` instance can be passed in as the `learning_rate`
    argument of any optimizer.
    To implement your own schedule object, you should implement the `__call__`
    method, which takes a `step` argument (scalar integer tensor, the
    current training step count).
    Like for any other Keras object, you can also optionally
    make your object serializable by implementing the `get_config`
    and `from_config` methods.
    Example:
    ```python
    class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
      def __call__(self, step):
         return self.initial_learning_rate / (step + 1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
    ```
    """
    @abc.abstractmethod
    def __call__(self, step):
        raise NotImplementedError(
            f"Learning rate schedule '{self.__class__.__name__}' "
            "must override `__call__(self, step)`."
        )
    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError(
            f"Learning rate schedule '{self.__class__.__name__}' "
            "must override `get_config()` in order to be serializable."
        )
    @classmethod
    def from_config(cls, config):
        """Instantiates a `LearningRateSchedule` from its config.
        Args:
            config: Output of `get_config()`.
        Returns:
            A `LearningRateSchedule` instance.
        """
        return cls(**config)
