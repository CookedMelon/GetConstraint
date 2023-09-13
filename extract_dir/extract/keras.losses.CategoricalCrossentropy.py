@keras_export("keras.losses.CategoricalCrossentropy")
class CategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.
    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a `one_hot` representation. If
    you want to provide labels as integers, please use
    `SparseCategoricalCrossentropy` loss.  There should be `# classes` floating
    point values per feature.
    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.
    Standalone usage:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred).numpy()
    1.177
    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.814
    >>> # Using 'sum' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cce(y_true, y_pred).numpy()
    2.354
    >>> # Using 'none' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cce(y_true, y_pred).numpy()
    array([0.0513, 2.303], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CategoricalCrossentropy())
    ```
    """
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="categorical_crossentropy",
    ):
        """Initializes `CategoricalCrossentropy` instance.
        Args:
          from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
          label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
          axis: The axis along which to compute crossentropy (the features
            axis). Defaults to -1.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: Optional name for the instance.
            Defaults to 'categorical_crossentropy'.
        """
        super().__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
