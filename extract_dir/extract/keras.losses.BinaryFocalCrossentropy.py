@keras_export("keras.losses.BinaryFocalCrossentropy")
class BinaryFocalCrossentropy(LossFunctionWrapper):
    """Computes focal cross-entropy loss between true labels and predictions.
    Binary cross-entropy loss is often used for binary (0 or 1) classification
    tasks. The loss function requires the following inputs:
    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
      floating-point value which either represents a
      [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
      when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
      `from_logits=False`).
    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a "focal factor" to down-weight easy examples and focus more
    on hard examples. By default, the focal tensor is computed as follows:
    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma=0`, this function is
    equivalent to the binary crossentropy loss.
    With the `compile()` API:
    ```python
    model.compile(
      loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True),
      ....
    )
    ```
    As a standalone function:
    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = [0, 1, 0, 0]
    >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2,
    ...                                                from_logits=True)
    >>> loss(y_true, y_pred).numpy()
    0.691
    >>> # Apply class weight
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=2, from_logits=True)
    >>> loss(y_true, y_pred).numpy()
    0.51
    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=3,
    ...                                                from_logits=True)
    >>> loss(y_true, y_pred).numpy()
    0.647
    >>> # Apply class weight
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred).numpy()
    0.482
    >>> # Using 'sample_weight' attribute with focal effect
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=3,
    ...                                                from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
    0.133
    >>> # Apply class weight
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
    0.097
    >>> # Using 'sum' reduction` type.
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=4,
    ...                                                from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> loss(y_true, y_pred).numpy()
    1.222
    >>> # Apply class weight
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=4, from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> loss(y_true, y_pred).numpy()
    0.914
    >>> # Using 'none' reduction type.
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     gamma=5, from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> loss(y_true, y_pred).numpy()
    array([0.0017 1.1561], dtype=float32)
    >>> # Apply class weight
    >>> loss = tf.keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=5, from_logits=True,
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> loss(y_true, y_pred).numpy()
    array([0.0004 0.8670], dtype=float32)
    Args:
      apply_class_balancing: A bool, whether to apply weight balancing on the
        binary classes 0 and 1.
      alpha: A weight balancing factor for class 1, default is `0.25` as
        mentioned in reference [Lin et al., 2018](
        https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
        `1.0 - alpha`.
      gamma: A focusing parameter used to compute the focal factor, default is
        `2.0` as mentioned in the reference
        [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
      from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
        assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
      label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs. When >
        `0`, we compute the loss between the predicted labels and a smoothed
        version of the true labels, where the smoothing squeezes the labels
        towards `0.5`. Larger values of `label_smoothing` correspond to heavier
        smoothing.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to `-1`.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
        `tf.distribute.Strategy`, except via `Model.compile()` and
        `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
        https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Name for the op. Defaults to 'binary_focal_crossentropy'.
    """
    def __init__(
        self,
        apply_class_balancing=False,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="binary_focal_crossentropy",
    ):
        """Initializes `BinaryFocalCrossentropy` instance."""
        super().__init__(
            binary_focal_crossentropy,
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            gamma=gamma,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma
    def get_config(self):
        config = {
            "apply_class_balancing": self.apply_class_balancing,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
