@keras_export("keras.metrics.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(base_metric.MeanMetricWrapper):
    """Computes the mean absolute percentage error between `y_true` and
    `y_pred`.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.MeanAbsolutePercentageError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    250000000.0
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    500000000.0
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_absolute_percentage_error", dtype=None):
        super().__init__(mean_absolute_percentage_error, name, dtype=dtype)
