@keras_export("keras.metrics.SparseTopKCategoricalAccuracy")
class SparseTopKCategoricalAccuracy(base_metric.MeanMetricWrapper):
    """Computes how often integer targets are in the top `K` predictions.
    Args:
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5
    >>> m.reset_state()
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None
    ):
        super().__init__(
            metrics_utils.sparse_top_k_categorical_matches,
            name,
            dtype=dtype,
            k=k,
        )
