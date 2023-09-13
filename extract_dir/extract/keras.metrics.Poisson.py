@keras_export("keras.metrics.Poisson")
class Poisson(base_metric.MeanMetricWrapper):
    """Computes the Poisson score between `y_true` and `y_pred`.
    🐟 🐟 🐟
    It is defined as: `poisson_score = y_pred - y_true * log(y_pred)`.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.Poisson()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.49999997
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.99999994
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Poisson()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="poisson", dtype=None):
        super().__init__(poisson, name, dtype=dtype)
