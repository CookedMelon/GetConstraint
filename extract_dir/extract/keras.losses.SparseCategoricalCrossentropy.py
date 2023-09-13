@keras_export("keras.losses.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.
    Use this crossentropy loss function when there are two or more label
    classes.  We expect labels to be provided as integers. If you want to
    provide labels using `one-hot` representation, please use
    `CategoricalCrossentropy` loss.  There should be `# classes` floating point
    values per feature for `y_pred` and a single floating point value per
    feature for `y_true`.
    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.
    Standalone usage:
    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
    >>> scce(y_true, y_pred).numpy()
    1.177
    >>> # Calling with 'sample_weight'.
    >>> scce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.814
    >>> # Using 'sum' reduction type.
    >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> scce(y_true, y_pred).numpy()
    2.354
    >>> # Using 'none' reduction type.
    >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> scce(y_true, y_pred).numpy()
    array([0.0513, 2.303], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    ```
    """
    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction=losses_utils.ReductionV2.AUTO,
        name="sparse_categorical_crossentropy",
    ):
        """Initializes `SparseCategoricalCrossentropy` instance.
        Args:
          from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
          ignore_class: Optional integer. The ID of a class to be ignored during
            loss computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps.
            By default (`ignore_class=None`), all classes are considered.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: Optional name for the instance. Defaults to
            'sparse_categorical_crossentropy'.
        """
        super().__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            ignore_class=ignore_class,
        )
