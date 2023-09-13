@keras_export("keras.losses.SquaredHinge")
class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` & `y_pred`.
    `loss = square(maximum(1 - y_true * y_pred, 0))`
    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.
    Standalone usage:
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> h = tf.keras.losses.SquaredHinge()
    >>> h(y_true, y_pred).numpy()
    1.86
    >>> # Calling with 'sample_weight'.
    >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
    0.73
    >>> # Using 'sum' reduction type.
    >>> h = tf.keras.losses.SquaredHinge(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> h(y_true, y_pred).numpy()
    3.72
    >>> # Using 'none' reduction type.
    >>> h = tf.keras.losses.SquaredHinge(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> h(y_true, y_pred).numpy()
    array([1.46, 2.26], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd', loss=tf.keras.losses.SquaredHinge())
    ```
    """
    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="squared_hinge"
    ):
        """Initializes `SquaredHinge` instance.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: Optional name for the instance. Defaults to 'squared_hinge'.
        """
        super().__init__(squared_hinge, name=name, reduction=reduction)
