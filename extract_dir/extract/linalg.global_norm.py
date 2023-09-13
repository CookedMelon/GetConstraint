@tf_export("linalg.global_norm", v1=["linalg.global_norm", "global_norm"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("global_norm")
def global_norm(t_list, name=None):
  """Computes the global norm of multiple tensors.
  Given a tuple or list of tensors `t_list`, this operation returns the
  global norm of the elements in all tensors in `t_list`. The global norm is
  computed as:
  `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`
  Any entries in `t_list` that are of type None are ignored.
  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    name: A name for the operation (optional).
  Returns:
    A 0-D (scalar) `Tensor` of type `float`.
  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  if (not isinstance(t_list, collections_abc.Sequence) or
      isinstance(t_list, str)):
    raise TypeError("`t_list` should be a sequence of tensors. Received "
                    f"{type(t_list)}.")
  t_list = list(t_list)
  with ops.name_scope(name, "global_norm", t_list) as name:
    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
            name="t_%d" % i) if t is not None else t
        for i, t in enumerate(t_list)
    ]
    half_squared_norms = []
    for v in values:
      if v is not None:
        with ops.colocate_with(v):
          half_squared_norms.append(gen_nn_ops.l2_loss(v))
    half_squared_norm = math_ops.reduce_sum(
        array_ops_stack.stack(half_squared_norms))
    norm = math_ops.sqrt(
        half_squared_norm *
        constant_op.constant(2.0, dtype=half_squared_norm.dtype),
        name="global_norm")
  return norm
