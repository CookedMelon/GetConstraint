@keras_export("keras.metrics.Accuracy")
class Accuracy(base_metric.MeanMetricWrapper):
    """Calculates how often predictions equal labels.
    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.Accuracy()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
    >>> m.result().numpy()
    0.75
    >>> m.reset_state()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
    ...                sample_weight=[1, 1, 0, 0])
    >>> m.result().numpy()
    0.5
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Accuracy()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="accuracy", dtype=None):
        super().__init__(accuracy, name, dtype=dtype)
