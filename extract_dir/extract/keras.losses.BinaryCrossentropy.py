@keras_export("keras.losses.BinaryCrossentropy")
class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.
    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:
    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
      floating-point value which either represents a
      [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
      when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
      `from_logits=False`).
    **Recommended Usage:** (set `from_logits=True`)
    With `tf.keras` API:
    ```python
    model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      ....
    )
    ```
    As a standalone function:
    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = [0, 1, 0, 0]
    >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
    >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred).numpy()
    0.865
    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred).numpy()
    0.865
    >>> # Using 'sample_weight' attribute
    >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
    0.243
    >>> # Using 'sum' reduction` type.
    >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> bce(y_true, y_pred).numpy()
    1.730
    >>> # Using 'none' reduction type.
    >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> bce(y_true, y_pred).numpy()
    array([0.235, 1.496], dtype=float32)
    **Default Usage:** (set `from_logits=False`)
    >>> # Make the following updates to the above "Recommended Usage" section
    >>> # 1. Set `from_logits=False`
    >>> tf.keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
    >>> # 2. Update `y_pred` to use probabilities instead of logits
    >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
    """
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="binary_crossentropy",
    ):
        """Initializes `BinaryCrossentropy` instance.
        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.
          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: Name for the op. Defaults to 'binary_crossentropy'.
        """
        super().__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
