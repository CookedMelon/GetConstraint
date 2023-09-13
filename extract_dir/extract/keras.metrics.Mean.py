@keras_export("keras.metrics.Mean")
class Mean(Reduce):
    """Computes the (weighted) mean of the given values.
    For example, if values is [1, 3, 5, 7] then the mean is 4.
    If the weights were specified as [1, 1, 0, 0] then the mean would be 2.
    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as
    `mean` which is an idempotent operation that simply divides `total` by
    `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.Mean()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result().numpy()
    4.0
    >>> m.reset_state()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result().numpy()
    2.0
    Usage with `compile()` API:
    ```python
    model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
    model.compile(optimizer='sgd', loss='mse')
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="mean", dtype=None):
        super().__init__(
            reduction=metrics_utils.Reduction.WEIGHTED_MEAN,
            name=name,
            dtype=dtype,
        )
