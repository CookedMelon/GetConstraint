"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.mean_squared_error",
    "keras.metrics.mse",
    "keras.metrics.MSE",
    "keras.losses.mean_squared_error",
    "keras.losses.mse",
    "keras.losses.MSE",
)
@tf.__internal__.dispatch.add_dispatch_support
def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.
    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.
    `loss = mean(square(y_true - y_pred), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)
def _ragged_tensor_apply_loss(loss_fn, y_true, y_pred, y_pred_extra_dim=False):
    """Apply a loss function on a per batch basis.
    Args:
      loss_fn: The loss function
      y_true: truth values (RaggedTensor)
      y_pred: predicted values (RaggedTensor)
      y_pred_extra_dim: whether y_pred has an additional dimension compared to
        y_true
    Returns:
      Loss-function result. A dense tensor if the output has a single dimension
      (per-batch loss value); a ragged tensor otherwise.
    """
    def rt_is_equiv_dense(rt):
        """Returns true if this RaggedTensor has the same row_lengths across
           all ragged dimensions and thus can be converted to a dense tensor
           without loss of information.
        Args:
          rt: RaggedTensor.
        """
        return tf.reduce_all(
            [
                tf.equal(
                    tf.math.reduce_variance(
                        tf.cast(row_lens, backend.floatx())
                    ),
                    tf.constant([0.0]),
                )
                for row_lens in rt.nested_row_lengths()
            ]
        )
    def _convert_to_dense(inputs):
        return tuple(
            rt.to_tensor() if isinstance(rt, tf.RaggedTensor) else rt
            for rt in inputs
        )
    def _call_loss(inputs, ragged_output):
        """Adapt the result to ragged or dense tensor according to the expected
        output type. This is done so that all the return values of the map
        operation have the same type.
        """
        r = loss_fn(*inputs)
        if ragged_output and not isinstance(r, tf.RaggedTensor):
            r = tf.RaggedTensor.from_tensor(r)
        elif not ragged_output and isinstance(r, tf.RaggedTensor):
            r = r.to_tensor()
        return r
    def _wrapper(inputs, ragged_output):
        _, y_pred = inputs
        if isinstance(y_pred, tf.RaggedTensor):
            return tf.cond(
                rt_is_equiv_dense(y_pred),
                lambda: _call_loss(_convert_to_dense(inputs), ragged_output),
                lambda: _call_loss(inputs, ragged_output),
            )
        return loss_fn(*inputs)
    if not isinstance(y_true, tf.RaggedTensor):
        return loss_fn(y_true, y_pred.to_tensor())
    lshape = y_pred.shape.as_list()[1:-1]
    if len(lshape) > 0:
        spec = tf.RaggedTensorSpec(shape=lshape, dtype=y_pred.dtype)
    else:
        spec = tf.TensorSpec(shape=[], dtype=y_pred.dtype)
    nested_splits_list = [rt.nested_row_splits for rt in (y_true, y_pred)]
    if y_pred_extra_dim:
        # The last dimension of a categorical prediction may be ragged or not.
        rdims = [len(slist) for slist in nested_splits_list]
        if rdims[0] == rdims[1] - 1:
            nested_splits_list[1] = nested_splits_list[1][:-1]
    map_fn = functools.partial(_wrapper, ragged_output=len(lshape) > 1)
    assertion_list = ragged_util.assert_splits_match(nested_splits_list)
    with tf.control_dependencies(assertion_list):
        return ragged_map_ops.map_fn(map_fn, elems=(y_true, y_pred), dtype=spec)
@dispatch.dispatch_for_types(mean_squared_error, tf.RaggedTensor)
def _ragged_tensor_mse(y_true, y_pred):
    """Implements support for handling RaggedTensors.
    Args:
      y_true: RaggedTensor truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: RaggedTensor predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
      When the number of dimensions of the batch feature vector [d0, .. dN] is
      greater than one the return value is a RaggedTensor. Otherwise a Dense
      tensor with dimensions [batch_size] is returned.
    """
    return _ragged_tensor_apply_loss(mean_squared_error, y_true, y_pred)
