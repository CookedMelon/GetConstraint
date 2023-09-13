@keras_export("keras.metrics.MeanAbsoluteError")
class MeanAbsoluteError(base_metric.MeanMetricWrapper):
    """Computes the mean absolute error between the labels and predictions.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.MeanAbsoluteError()
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
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_absolute_error", dtype=None):
        super().__init__(mean_absolute_error, name, dtype=dtype)
