@keras_export("keras.losses.CosineSimilarity")
class CosineSimilarity(LossFunctionWrapper):
    """Computes the cosine similarity between labels and predictions.
    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. The values closer to 1 indicate greater
    dissimilarity. This makes it usable as a loss function in a setting
    where you try to maximize the proximity between predictions and targets.
    If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0
    regardless of the proximity between predictions and targets.
    `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`
    Standalone usage:
    >>> y_true = [[0., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> cosine_loss(y_true, y_pred).numpy()
    -0.5
    >>> # Calling with 'sample_weight'.
    >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
    -0.0999
    >>> # Using 'sum' reduction type.
    >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cosine_loss(y_true, y_pred).numpy()
    -0.999
    >>> # Using 'none' reduction type.
    >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cosine_loss(y_true, y_pred).numpy()
    array([-0., -0.999], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CosineSimilarity(axis=1))
    ```
    Args:
      axis: The axis along which the cosine similarity is computed
        (the features axis). Defaults to -1.
      reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
        Default value is `AUTO`. `AUTO` indicates that the reduction option will
        be determined by the usage context. For almost all cases this defaults
        to `SUM_OVER_BATCH_SIZE`. When used under a
        `tf.distribute.Strategy`, except via `Model.compile()` and
        `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
        https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the instance.
    """
    def __init__(
        self,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="cosine_similarity",
    ):
        super().__init__(
            cosine_similarity, reduction=reduction, name=name, axis=axis
        )
