@tf_export("math.argmax", "argmax", v1=[])
@dispatch.add_dispatch_support
def argmax_v2(input, axis=None, output_type=dtypes.int64, name=None):
  """Returns the index with the largest value across axes of a tensor.
  In case of identity returns the smallest index.
  For example:
  >>> A = tf.constant([2, 20, 30, 3, 6])
  >>> tf.math.argmax(A)  # A[2] is maximum in tensor A
  <tf.Tensor: shape=(), dtype=int64, numpy=2>
  >>> B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
  ...                  [14, 45, 23, 5, 27]])
  >>> tf.math.argmax(B, 0)
  <tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2])>
  >>> tf.math.argmax(B, 1)
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 1])>
  >>> C = tf.constant([0, 0, 0, 0])
  >>> tf.math.argmax(C) # Returns smallest index in case of ties
  <tf.Tensor: shape=(), dtype=int64, numpy=0>
  Args:
    input: A `Tensor`.
    axis: An integer, the axis to reduce across. Default to 0.
    output_type: An optional output dtype (`tf.int32` or `tf.int64`). Defaults
      to `tf.int64`.
    name: An optional name for the operation.
  Returns:
    A `Tensor` of type `output_type`.
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_max(input, axis, name=name, output_type=output_type)
