@keras_export("keras.metrics.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(base_metric.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.
    Use this crossentropy metric when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` metric.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.
    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      axis: (Optional) Defaults to -1. The dimension along which entropy is
        computed.
    Standalone usage:
    >>> # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    >>> # logits = log(y_pred)
    >>> # softmax = exp(logits) / sum(exp(logits), axis=-1)
    >>> # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(softmax), 1)
    >>> # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    >>> #                [-2.3026, -0.2231, -2.3026]]
    >>> # y_true * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    >>> # xent = [0.0513, 2.3026]
    >>> # Reduced xent = (0.0513 + 2.3026) / 2
    >>> m = tf.keras.metrics.SparseCategoricalCrossentropy()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result().numpy()
    1.1769392
    >>> m.reset_state()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=tf.constant([0.3, 0.7]))
    >>> m.result().numpy()
    1.6271976
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        name: str = "sparse_categorical_crossentropy",
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        from_logits: bool = False,
        ignore_class: Optional[int] = None,
        axis: int = -1,
    ):
        super().__init__(
            sparse_categorical_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
        )
