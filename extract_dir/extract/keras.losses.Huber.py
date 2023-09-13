@keras_export("keras.losses.Huber")
class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` & `y_pred`.
    For each value x in `error = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
    Standalone usage:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> h = tf.keras.losses.Huber()
    >>> h(y_true, y_pred).numpy()
    0.155
    >>> # Calling with 'sample_weight'.
    >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
    0.09
    >>> # Using 'sum' reduction type.
    >>> h = tf.keras.losses.Huber(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> h(y_true, y_pred).numpy()
    0.31
    >>> # Using 'none' reduction type.
    >>> h = tf.keras.losses.Huber(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> h(y_true, y_pred).numpy()
    array([0.18, 0.13], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
    ```
    """
    def __init__(
        self,
        delta=1.0,
        reduction=losses_utils.ReductionV2.AUTO,
        name="huber_loss",
    ):
        """Initializes `Huber` instance.
        Args:
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: Optional name for the instance. Defaults to 'huber_loss'.
        """
        super().__init__(huber, name=name, reduction=reduction, delta=delta)
