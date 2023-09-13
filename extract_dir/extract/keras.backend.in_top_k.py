@keras_export("keras.backend.in_top_k")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`.
    Args:
        predictions: A tensor of shape `(batch_size, classes)` and type
          `float32`.
        targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.
    Returns:
        A 1D tensor of length `batch_size` and type `bool`.
        `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
        values of `predictions[i]`.
    """
    return tf.compat.v1.math.in_top_k(predictions, targets, k)
