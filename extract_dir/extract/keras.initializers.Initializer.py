@keras_export("keras.initializers.Initializer")
class Initializer:
    """Initializer base class: all Keras initializers inherit from this class.
    Initializers should implement a `__call__()` method with the following
    signature:
    ```python
    def __call__(self, shape, dtype=None, **kwargs):
        # returns a tensor of shape `shape` and dtype `dtype`
        # containing values drawn from a distribution of your choice.
        return tf.random.uniform(shape=shape, dtype=dtype)
    ```
    Optionally, you an also implement the method `get_config()` and the class
    method `from_config()` in order to support serialization -- just like with
    any Keras object.
    Here's a simple example: a random normal initializer.
    ```python
    class ExampleRandomNormal(Initializer):
        def __init__(self, mean, stddev):
            self.mean = mean
            self.stddev = stddev
        def __call__(self, shape, dtype=None, **kwargs):
            return tf.random.normal(
                shape, mean=self.mean, stddev=self.stddev, dtype=dtype
            )
        def get_config(self):  # To support serialization
            return {"mean": self.mean, "stddev": self.stddev}
    ```
    Note that we don't have to implement `from_config()` in the example above
    since the constructor arguments of the class the keys in the config returned
    by `get_config` are the same. In this case, the default `from_config()`
    works fine.
    """
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor.
          **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )
    def get_config(self):
        """Returns the initializer's configuration as a JSON-serializable dict.
        Returns:
            A JSON-serializable Python dict.
        """
        return {}
    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.
        Example:
        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```
        Args:
            config: A Python dictionary, the output of `get_config()`.
        Returns:
            An `Initializer` instance.
        """
        config.pop("dtype", None)
        return cls(**config)
    def _warn_reuse(self):
        if getattr(self, "_used", False):
            if getattr(self, "seed", None) is None:
                warnings.warn(
                    f"The initializer {self.__class__.__name__} is unseeded "
                    "and being called multiple times, which will return "
                    "identical values each time (even if the initializer is "
                    "unseeded). Please update your code to provide a seed to "
                    "the initializer, or avoid using the same initializer "
                    "instance more than once."
                )
        else:
            self._used = True
