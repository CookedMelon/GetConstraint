@tf_export("is_tensor")
def is_tf_type(x):  # pylint: disable=invalid-name
  """Checks whether `x` is a TF-native type that can be passed to many TF ops.
  Use `is_tensor` to differentiate types that can ingested by TensorFlow ops
  without any conversion (e.g., `tf.Tensor`, `tf.SparseTensor`, and
  `tf.RaggedTensor`) from types that need to be converted into tensors before
  they are ingested (e.g., numpy `ndarray` and Python scalars).
  For example, in the following code block:
  ```python
  if not tf.is_tensor(t):
    t = tf.convert_to_tensor(t)
  return t.shape, t.dtype
  ```
  we check to make sure that `t` is a tensor (and convert it if not) before
  accessing its `shape` and `dtype`.  (But note that not all TensorFlow native
  types have shapes or dtypes; `tf.data.Dataset` is an example of a TensorFlow
  native type that has neither shape nor dtype.)
  Args:
    x: A python object to check.
  Returns:
    `True` if `x` is a TensorFlow-native type.
  """
  return isinstance(x, tf_type_classes)
