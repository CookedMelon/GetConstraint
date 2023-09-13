@tf_export("math.count_nonzero", v1=[])
@dispatch.add_dispatch_support
def count_nonzero_v2(
    input,  # pylint: disable=redefined-builtin
    axis=None,
    keepdims=None,
    dtype=dtypes.int64,
    name=None):
  """Computes number of nonzero elements across dimensions of a tensor.
  Reduces `input` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.
  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
  **NOTE** Floating point comparison to zero is done by exact floating point
  equality check.  Small values are **not** rounded to zero for purposes of
  the nonzero check.
  For example:
  ```python
  x = tf.constant([[0, 1, 0], [1, 1, 0]])
  tf.math.count_nonzero(x)  # 3
  tf.math.count_nonzero(x, 0)  # [1, 2, 0]
  tf.math.count_nonzero(x, 1)  # [1, 2]
  tf.math.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.math.count_nonzero(x, [0, 1])  # 3
  ```
  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.
  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.math.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```
  Args:
    input: The tensor to reduce. Should be of numeric type, `bool`, or `string`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input), rank(input))`.
    keepdims: If true, retains reduced dimensions with length 1.
    dtype: The output dtype; defaults to `tf.int64`.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor (number of nonzero values).
  """
  if keepdims is None:
    keepdims = False
  with ops.name_scope(name, "count_nonzero", [input]):
    input = ops.convert_to_tensor(input, name="input")
    # A scalar of 'zero' is enough as `not_equal` will broadcast.
    zero = array_ops.zeros([], dtype=input.dtype)
    return cast(
        reduce_sum(
            # int64 reduction happens on GPU
            cast(gen_math_ops.not_equal(input, zero), dtypes.int64),
            axis=axis,
            keepdims=keepdims),
        dtype=dtype)
