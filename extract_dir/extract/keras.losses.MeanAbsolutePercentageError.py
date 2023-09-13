@keras_export("keras.losses.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.
    Formula:
    `loss = 100 * abs((y_true - y_pred) / y_true)`
    Note that to avoid dividing by zero, a small epsilon value
    is added to the denominator.
    Standalone usage:
    >>> y_true = [[2., 1.], [2., 3.]]
    >>> y_pred = [[1., 1.], [1., 0.]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> mape = tf.keras.losses.MeanAbsolutePercentageError()
    >>> mape(y_true, y_pred).numpy()
    50.
    >>> # Calling with 'sample_weight'.
    >>> mape(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
    20.
    >>> # Using 'sum' reduction type.
    >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> mape(y_true, y_pred).numpy()
    100.
    >>> # Using 'none' reduction type.
    >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> mape(y_true, y_pred).numpy()
    array([25., 75.], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.MeanAbsolutePercentageError())
    ```
    """
    def __init__(
        self,
        reduction=losses_utils.ReductionV2.AUTO,
        name="mean_absolute_percentage_error",
    ):
        """Initializes `MeanAbsolutePercentageError` instance.
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
          name: Optional name for the instance. Defaults to
            'mean_absolute_percentage_error'.
        """
        super().__init__(
            mean_absolute_percentage_error, name=name, reduction=reduction
        )
