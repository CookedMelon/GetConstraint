@tf_export("nn.ctc_unique_labels")
@dispatch.add_dispatch_support
def ctc_unique_labels(labels, name=None):
  """Get unique labels and indices for batched labels for `tf.nn.ctc_loss`.
  For use with `tf.nn.ctc_loss` optional argument `unique`: This op can be
  used to preprocess labels in input pipeline to for better speed/memory use
  computing the ctc loss on TPU.
  Example:
    ctc_unique_labels([[3, 4, 4, 3]]) ->
      unique labels padded with 0: [[3, 4, 0, 0]]
      indices of original labels in unique: [0, 1, 1, 0]
  Args:
    labels: tensor of shape [batch_size, max_label_length] padded with 0.
    name: A name for this `Op`. Defaults to "ctc_unique_labels".
  Returns:
    tuple of
      - unique labels, tensor of shape `[batch_size, max_label_length]`
      - indices into unique labels, shape `[batch_size, max_label_length]`
  """
  with ops.name_scope(name, "ctc_unique_labels", [labels]):
    labels = ops.convert_to_tensor(labels, name="labels")
    def _unique(x):
      u = array_ops.unique(x)
      y = array_ops.pad(u.y, [[0, _get_dim(u.idx, 0) - _get_dim(u.y, 0)]])
      y = math_ops.cast(y, dtypes.int64)
      return [y, u.idx]
    return map_fn.map_fn(_unique, labels, dtype=[dtypes.int64, dtypes.int32])
