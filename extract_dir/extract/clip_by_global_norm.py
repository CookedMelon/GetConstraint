@tf_export("clip_by_global_norm")
@dispatch.add_dispatch_support
def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.
  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.
  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`
  to signal that an error occurred.
  Any of the entries of `t_list` that are of type `None` are ignored.
  This is the correct way to perform gradient clipping (Pascanu et al., 2012).
  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.
  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).
  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.
  Raises:
    TypeError: If `t_list` is not a sequence.
  References:
    On the difficulty of training Recurrent Neural Networks:
      [Pascanu et al., 2012](http://proceedings.mlr.press/v28/pascanu13.html)
      ([pdf](http://proceedings.mlr.press/v28/pascanu13.pdf))
  """
  if (not isinstance(t_list, collections_abc.Sequence) or
      isinstance(t_list, str)):
    raise TypeError("`t_list` should be a sequence of tensors. Received "
                    f"{type(t_list)}.")
  t_list = list(t_list)
  if use_norm is None:
    use_norm = global_norm(t_list, name)
  with ops.name_scope(name, "clip_by_global_norm",
                      t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale_for_finite = clip_norm * math_ops.minimum(
        1.0 / use_norm,
        constant_op.constant(1.0, dtype=use_norm.dtype) / clip_norm)
    # If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
    # this will make scale NaN.
    scale = scale_for_finite + (use_norm - use_norm)
    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
            name="t_%d" % i) if t is not None else t
        for i, t in enumerate(t_list)
    ]
    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with ops.colocate_with(v):
          values_clipped.append(
              array_ops.identity(v * scale, name="%s_%d" % (name, i)))
    list_clipped = [
        indexed_slices.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, indexed_slices.IndexedSlices) else c_v
        for (c_v, t) in zip(values_clipped, t_list)
    ]
  return list_clipped, use_norm
