@tf_export("concat")
@dispatch.add_dispatch_support
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.
  See also `tf.tile`, `tf.stack`, `tf.repeat`.
  Concatenates the list of tensors `values` along dimension `axis`.  If
  `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
  result has shape
      [D0, D1, ... Raxis, ...Dn]
  where
      Raxis = sum(Daxis(i))
  That is, the data from the input tensors is joined along the `axis`
  dimension.
  The number of dimensions of the input tensors must match, and all dimensions
  except `axis` must be equal.
  For example:
  >>> t1 = [[1, 2, 3], [4, 5, 6]]
  >>> t2 = [[7, 8, 9], [10, 11, 12]]
  >>> tf.concat([t1, t2], 0)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12]], dtype=int32)>
  >>> tf.concat([t1, t2], 1)
  <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
  array([[ 1,  2,  3,  7,  8,  9],
         [ 4,  5,  6, 10, 11, 12]], dtype=int32)>
  As in Python, the `axis` could also be negative numbers. Negative `axis`
  are interpreted as counting from the end of the rank, i.e.,
   `axis + rank(values)`-th dimension.
  For example:
  >>> t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
  >>> t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
  >>> tf.concat([t1, t2], -1)
  <tf.Tensor: shape=(2, 2, 4), dtype=int32, numpy=
    array([[[ 1,  2,  7,  4],
            [ 2,  3,  8,  4]],
           [[ 4,  4,  2, 10],
            [ 5,  3, 15, 11]]], dtype=int32)>
  Note: If you are concatenating along a new axis consider using stack.
  E.g.
  ```python
  tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
  ```
  can be rewritten as
  ```python
  tf.stack(tensors, axis=axis)
  ```
  Args:
    values: A list of `Tensor` objects or a single `Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` resulting from concatenation of the input tensors.
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  # TODO(mrry): Change to return values?
  if len(values) == 1:  # Degenerate case of one tensor.
    # Make a throwaway call to convert_to_tensor to make sure
    # that axis is of the correct type, and make sure that
    # the returned tensor is a scalar.
    # TODO(keveman): Implement a standalone type and shape checker.
    with ops.name_scope(name) as scope:
      ops.convert_to_tensor(
          axis, name="concat_dim",
          dtype=dtypes.int32).get_shape().assert_has_rank(0)
      return identity(values[0], name=name)
  return gen_array_ops.concat_v2(values=values, axis=axis, name=name)
