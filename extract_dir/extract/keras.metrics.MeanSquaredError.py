@keras_export("keras.metrics.MeanSquaredError")
class MeanSquaredError(base_metric.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.25
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.5
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanSquaredError()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(mean_squared_error, name, dtype=dtype)
