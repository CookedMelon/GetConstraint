@tf_export("clip_by_norm")
@dispatch.add_dispatch_support
def clip_by_norm(t, clip_norm, axes=None, name=None):
  """Clips tensor values to a maximum L2-norm.
  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
  along the dimensions given in `axes`. Specifically, in the default case
  where all dimensions are used for calculation, if the L2-norm of `t` is
  already less than or equal to `clip_norm`, then `t` is not modified. If
  the L2-norm is greater than `clip_norm`, then this operation returns a
  tensor of the same type and shape as `t` with its values set to:
  `t * clip_norm / l2norm(t)`
  In this case, the L2-norm of the output tensor is `clip_norm`.
  As another example, if `t` is a matrix and `axes == [1]`, then each row
  of the output will have L2-norm less than or equal to `clip_norm`. If
  `axes == [0]` instead, each column of the output will be clipped.
  Code example:
  >>> some_nums = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)
  >>> tf.clip_by_norm(some_nums, 2.0).numpy()
  array([[0.26967996, 0.5393599 , 0.80903983, 1.0787199 , 1.3483998 ]],
        dtype=float32)
  This operation is typically used to clip gradients before applying them with
  an optimizer.  Most gradient data is a collection of different shaped tensors
  for different parts of the model.  Thus, this is a common usage:
  ```
  # Get your gradients after training
  loss_value, grads = grad(model, features, labels)
  # Apply some clipping
  grads = [tf.clip_by_norm(g, norm)
               for g in grads]
  # Continue on with training
  optimizer.apply_gradients(grads)
  ```
  Args:
    t: A `Tensor` or `IndexedSlices`.  This must be a floating point type.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value, also
      floating point
    axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions
      to use for computing the L2-norm. If `None` (the default), uses all
      dimensions.
    name: A name for the operation (optional).
  Returns:
    A clipped `Tensor` or `IndexedSlices`.
  Raises:
    ValueError: If the clip_norm tensor is not a 0-D scalar tensor.
    TypeError: If dtype of the input is not a floating point or
      complex type.
  """
  with ops.name_scope(name, "clip_by_norm", [t, clip_norm]) as name:
    values = ops.convert_to_tensor(
        t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
        name="t")
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    l2sum = math_ops.reduce_sum(values * values, axes, keepdims=True)
    pred = l2sum > 0
    # Two-tap tf.where trick to bypass NaN gradients
    l2sum_safe = array_ops.where(pred, l2sum, array_ops.ones_like(l2sum))
    l2norm = array_ops.where(pred, math_ops.sqrt(l2sum_safe), l2sum)
    intermediate = values * clip_norm
    # Assert that the shape is compatible with the initial shape,
    # to prevent unintentional broadcasting.
    values.shape.assert_is_compatible_with(intermediate.shape)
    values_clip = array_ops.identity(
        intermediate / math_ops.maximum(l2norm, clip_norm), name=name)
    if isinstance(t, indexed_slices.IndexedSlices):
      return indexed_slices.IndexedSlices(values_clip, t.indices, t.dense_shape)
    return values_clip
