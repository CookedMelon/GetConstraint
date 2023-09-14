@keras_export(
    "keras.metrics.sparse_categorical_crossentropy",
    "keras.losses.sparse_categorical_crossentropy",
)
@tf.__internal__.dispatch.add_dispatch_support
def sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, axis=-1, ignore_class=None
):
    """Computes the sparse categorical crossentropy loss.
    Standalone usage:
    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)
    >>> y_true = [[[ 0,  2],
    ...            [-1, -1]],
    ...           [[ 0,  2],
    ...            [-1, -1]]]
    >>> y_pred = [[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ...             [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]],
    ...           [[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
    ...            [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]]]
    >>> loss = tf.keras.losses.sparse_categorical_crossentropy(
    ...   y_true, y_pred, ignore_class=-1)
    >>> loss.numpy()
    array([[[2.3841855e-07, 2.3841855e-07],
            [0.0000000e+00, 0.0000000e+00]],
           [[2.3841855e-07, 6.9314730e-01],
            [0.0000000e+00, 0.0000000e+00]]], dtype=float32)
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      axis: Defaults to -1. The dimension along which the entropy is
        computed.
      ignore_class: Optional integer. The ID of a class to be ignored during
        loss computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
    Returns:
      Sparse categorical crossentropy loss value.
    """
    return backend.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=from_logits,
        ignore_class=ignore_class,
        axis=axis,
    )
@dispatch.dispatch_for_types(sparse_categorical_crossentropy, tf.RaggedTensor)
def _ragged_tensor_sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, axis=-1, ignore_class=None
):
    """Implements support for handling RaggedTensors.
    Expected y_pred shape: (batch, sequence_len, n_classes) with sequence_len
    being variable per batch.
    Return shape: (batch, sequence_len).
    When used by SparseCategoricalCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
    number of elements independent of the batch. E.g. if the RaggedTensor
    has 2 batches with [2, 1] values respectively, the resulting loss is
    the sum of the individual loss values divided by 3.
    """
    fn = functools.partial(
        sparse_categorical_crossentropy,
        from_logits=from_logits,
        ignore_class=ignore_class,
        axis=axis,
    )
    return _ragged_tensor_apply_loss(fn, y_true, y_pred, y_pred_extra_dim=True)
