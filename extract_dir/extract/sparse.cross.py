@tf_export("sparse.cross")
def sparse_cross(inputs, name=None, separator=None):
  """Generates sparse cross from a list of sparse and dense tensors.
  For example, if the inputs are
      * inputs[0]: SparseTensor with shape = [2, 2]
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
      * inputs[1]: SparseTensor with shape = [2, 1]
        [0, 0]: "d"
        [1, 0]: "e"
      * inputs[2]: Tensor [["f"], ["g"]]
  then the output will be:
      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"
  Customized separator "_Y_":
  >>> inp_0 = tf.constant([['a'], ['b']])
  >>> inp_1 = tf.constant([['c'], ['d']])
  >>> output = tf.sparse.cross([inp_0, inp_1], separator='_Y_')
  >>> output.values
  <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'a_Y_c', b'b_Y_d'],
    dtype=object)>
  Args:
    inputs: An iterable of `Tensor` or `SparseTensor`.
    name: Optional name for the op.
    separator: A string added between each string being joined. Defaults to
      '_X_'.
  Returns:
    A `SparseTensor` of type `string`.
  """
  if separator is None:
    separator = "_X_"
  separator = ops.convert_to_tensor(separator, dtypes.string)
  indices, values, shapes, dense_inputs = _sparse_cross_internal_v2(inputs)
  indices_out, values_out, shape_out = gen_sparse_ops.sparse_cross_v2(
      indices=indices,
      values=values,
      shapes=shapes,
      dense_inputs=dense_inputs,
      sep=separator,
      name=name)
  return sparse_tensor.SparseTensor(indices_out, values_out, shape_out)
