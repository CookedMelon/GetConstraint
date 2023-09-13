@tf_export(v1=["nn.softmax", "math.softmax"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def softmax(logits, axis=None, name=None, dim=None):
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    axis = -1
  return _wrap_2d_function(logits, gen_nn_ops.softmax, axis, name)
