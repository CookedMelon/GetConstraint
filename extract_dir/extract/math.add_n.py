@tf_export("math.add_n", "add_n")
@dispatch.add_dispatch_support(iterable_parameters=["inputs"])
def add_n(inputs, name=None):
  """Returns the element-wise sum of a list of tensors.
  All inputs in the list must have the same shape. This op does not
  [broadcast](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
  its inputs. If you need broadcasting, use `tf.math.add` (or the `+` operator)
  instead.
  For example:
  >>> a = tf.constant([[3, 5], [4, 8]])
  >>> b = tf.constant([[1, 6], [2, 9]])
  >>> tf.math.add_n([a, b, a]).numpy()
  array([[ 7, 16],
         [10, 25]], dtype=int32)
  See Also:
  * `tf.reduce_sum(inputs, axis=0)` - This performs the same mathematical
    operation, but `tf.add_n` may be more efficient because it sums the
    tensors directly. `reduce_sum` on the other hand calls
    `tf.convert_to_tensor` on the list of tensors, unnecessarily stacking them
    into a single tensor before summing.
  Args:
    inputs: A list of `tf.Tensor` or `tf.IndexedSlices` objects, each with the
      same shape and type. `tf.IndexedSlices` objects will be converted into
      dense tensors prior to adding.
    name: A name for the operation (optional).
  Returns:
    A `tf.Tensor` of the same shape and type as the elements of `inputs`.
  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
  if not inputs or not isinstance(inputs, collections_abc.Iterable):
    raise ValueError("Inputs must be an iterable of at least one "
                     "Tensor/IndexedSlices with the same dtype and shape.")
  inputs = indexed_slices.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(
      isinstance(x, (ops.Tensor, indexed_slices.IndexedSlices))
      for x in inputs):
    raise ValueError("Inputs must be an iterable of at least one "
                     "Tensor/IndexedSlices with the same dtype and shape.")
  if len(inputs) == 1:
    if isinstance(inputs[0], indexed_slices.IndexedSlices):
      values = ops.convert_to_tensor(inputs[0])
    else:
      values = inputs[0]
    if name:
      return array_ops.identity(values, name=name)
    return values
  return gen_math_ops.add_n(inputs, name=name)
