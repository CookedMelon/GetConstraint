@keras_export("keras.constraints.Constraint")
class Constraint:
    """Base class for weight constraints.
    A `Constraint` instance works like a stateless function.
    Users who subclass this
    class should override the `__call__` method, which takes a single
    weight parameter and return a projected version of that parameter
    (e.g. normalized or clipped). Constraints can be used with various Keras
    layers via the `kernel_constraint` or `bias_constraint` arguments.
    Here's a simple example of a non-negative weight constraint:
    >>> class NonNegative(tf.keras.constraints.Constraint):
    ...
    ...  def __call__(self, w):
    ...    return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
    >>> weight = tf.constant((-1.0, 1.0))
    >>> NonNegative()(weight)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.,  1.],
    dtype=float32)>
    >>> tf.keras.layers.Dense(4, kernel_constraint=NonNegative())
    """
    def __call__(self, w):
        """Applies the constraint to the input weight variable.
        By default, the inputs weight variable is not modified.
        Users should override this method to implement their own projection
        function.
        Args:
          w: Input weight variable.
        Returns:
          Projected variable (by default, returns unmodified inputs).
        """
        return w
    def get_config(self):
        """Returns a Python dict of the object config.
        A constraint config is a Python dictionary (JSON-serializable) that can
        be used to reinstantiate the same object.
        Returns:
          Python dict containing the configuration of the constraint object.
        """
        return {}
    @classmethod
    def from_config(cls, config):
        """Instantiates a weight constraint from a configuration dictionary.
        Example:
        ```python
        constraint = UnitNorm()
        config = constraint.get_config()
        constraint = UnitNorm.from_config(config)
        ```
        Args:
          config: A Python dictionary, the output of `get_config`.
        Returns:
          A `tf.keras.constraints.Constraint` instance.
        """
        return cls(**config)
