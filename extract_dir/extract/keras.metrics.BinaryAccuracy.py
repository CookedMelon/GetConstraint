@keras_export("keras.metrics.BinaryAccuracy")
class BinaryAccuracy(base_metric.MeanMetricWrapper):
    """Calculates how often predictions match binary labels.
    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      threshold: (Optional) Float representing the threshold for deciding
      whether prediction values are 1 or 0.
    Standalone usage:
    >>> m = tf.keras.metrics.BinaryAccuracy()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
    >>> m.result().numpy()
    0.75
    >>> m.reset_state()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    0.5
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5):
        super().__init__(
            metrics_utils.binary_matches, name, dtype=dtype, threshold=threshold
        )
