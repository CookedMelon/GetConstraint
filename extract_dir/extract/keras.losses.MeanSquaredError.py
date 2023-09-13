@keras_export("keras.losses.MeanSquaredError")
class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.
    `loss = mean(square(y_true - y_pred))`
    Standalone usage:
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [1., 0.]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> mse = tf.keras.losses.MeanSquaredError()
    >>> mse(y_true, y_pred).numpy()
    0.5
    >>> # Calling with 'sample_weight'.
    >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
    0.25
    >>> # Using 'sum' reduction type.
    >>> mse = tf.keras.losses.MeanSquaredError(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> mse(y_true, y_pred).numpy()
    1.0
    >>> # Using 'none' reduction type.
    >>> mse = tf.keras.losses.MeanSquaredError(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> mse(y_true, y_pred).numpy()
    array([0.5, 0.5], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
    ```
    """
    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="mean_squared_error"
    ):
        """Initializes `MeanSquaredError` instance.
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
            'mean_squared_error'.
        """
        super().__init__(mean_squared_error, name=name, reduction=reduction)
