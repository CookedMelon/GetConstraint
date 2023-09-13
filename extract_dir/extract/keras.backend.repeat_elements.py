@keras_export("keras.backend.repeat_elements")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    Args:
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.
    Returns:
        A tensor.
    Example:
        >>> b = tf.constant([1, 2, 3])
        >>> tf.keras.backend.repeat_elements(b, rep=2, axis=0)
        <tf.Tensor: shape=(6,), dtype=int32,
            numpy=array([1, 1, 2, 2, 3, 3], dtype=int32)>
    """
    x_shape = x.shape.as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)
    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.
    # Repeating
    auxiliary_axis = axis + 1
    x_shape = tf.shape(x)
    x_rep = tf.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.shape) + 1)
    reps[auxiliary_axis] = rep
    x_rep = tf.tile(x_rep, reps)
    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = tf.constant(reps, dtype="int32")
    x_shape *= reps
    x_rep = tf.reshape(x_rep, x_shape)
    # Fix shape representation
    x_shape = x.shape.as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)
    return x_rep
