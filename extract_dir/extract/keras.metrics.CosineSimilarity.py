@keras_export("keras.metrics.CosineSimilarity")
class CosineSimilarity(base_metric.MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.
    `cosine similarity = (a . b) / ||a|| ||b||`
    See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      axis: (Optional) Defaults to -1. The dimension along which the cosine
        similarity is computed.
    Standalone usage:
    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> m = tf.keras.metrics.CosineSimilarity(axis=1)
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
    >>> m.result().numpy()
    0.49999997
    >>> m.reset_state()
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
    ...                sample_weight=[0.3, 0.7])
    >>> m.result().numpy()
    0.6999999
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1):
        super().__init__(cosine_similarity, name, dtype=dtype, axis=axis)
