@keras_export("keras.metrics.SparseCategoricalAccuracy")
class SparseCategoricalAccuracy(base_metric.MeanMetricWrapper):
    """Calculates how often predictions match integer labels.
    ```python
    acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    ```
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `sparse categorical accuracy`: an
    idempotent operation that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.SparseCategoricalAccuracy()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5
    >>> m.reset_state()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="sparse_categorical_accuracy", dtype=None):
        super().__init__(
            metrics_utils.sparse_categorical_matches, name, dtype=dtype
        )
