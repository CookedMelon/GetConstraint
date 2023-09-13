@tf_export("sparse.bincount")
def sparse_bincount(values,
                    weights=None,
                    axis=0,
                    minlength=None,
                    maxlength=None,
                    binary_output=False,
                    name=None):
  """Count the number of times an integer value appears in a tensor.
  This op takes an N-dimensional `Tensor`, `RaggedTensor`, or `SparseTensor`,
  and returns an N-dimensional int64 SparseTensor where element
  `[i0...i[axis], j]` contains the number of times the value `j` appears in
  slice `[i0...i[axis], :]` of the input tensor.  Currently, only N=0 and
  N=-1 are supported.
  Args:
    values: A Tensor, RaggedTensor, or SparseTensor whose values should be
      counted. These tensors must have a rank of 2 if `axis=-1`.
    weights: If non-None, must be the same shape as arr. For each value in
      `value`, the bin will be incremented by the corresponding weight instead
      of 1.
    axis: The axis to slice over. Axes at and below `axis` will be flattened
      before bin counting. Currently, only `0`, and `-1` are supported. If None,
      all axes will be flattened (identical to passing `0`).
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `values` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    binary_output: If True, this op will output 1 instead of the number of times
      a token appears (equivalent to one_hot + reduce_any instead of one_hot +
      reduce_add). Defaults to False.
    name: A name for this op.
  Returns:
    A SparseTensor with `output.shape = values.shape[:axis] + [N]`, where `N` is
      * `maxlength` (if set);
      * `minlength` (if set, and `minlength > reduce_max(values)`);
      * `0` (if `values` is empty);
      * `reduce_max(values) + 1` otherwise.
  Raises:
    `InvalidArgumentError` if negative values are provided as an input.
  Examples:
  **Bin-counting every item in individual batches**
  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i.
  >>> data = np.array([[10, 20, 30, 20], [11, 101, 11, 10001]], dtype=np.int64)
  >>> output = tf.sparse.bincount(data, axis=-1)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    0    30]
   [    1    11]
   [    1   101]
   [    1 10001]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([1 2 1 2 1 1], shape=(6,), dtype=int64),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))
  **Bin-counting with defined output shape**
  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i. However, all values of j
  above 'maxlength' are ignored. The dense_shape of the output sparse tensor
  is set to 'minlength'. Note that, while the input is identical to the
  example above, the value '10001' in batch item 2 is dropped, and the
  dense shape is [2, 500] instead of [2,10002] or [2, 102].
  >>> minlength = maxlength = 500
  >>> data = np.array([[10, 20, 30, 20], [11, 101, 11, 10001]], dtype=np.int64)
  >>> output = tf.sparse.bincount(
  ...    data, axis=-1, minlength=minlength, maxlength=maxlength)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[  0  10]
   [  0  20]
   [  0  30]
   [  1  11]
   [  1 101]], shape=(5, 2), dtype=int64),
   values=tf.Tensor([1 2 1 2 1], shape=(5,), dtype=int64),
   dense_shape=tf.Tensor([  2 500], shape=(2,), dtype=int64))
  **Binary bin-counting**
  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where (i,j) is 1 if the value j
  appears in batch i at least once and is 0 otherwise. Note that, even though
  some values (like 20 in batch 1 and 11 in batch 2) appear more than once,
  the 'values' tensor is all 1s.
  >>> data = np.array([[10, 20, 30, 20], [11, 101, 11, 10001]], dtype=np.int64)
  >>> output = tf.sparse.bincount(data, binary_output=True, axis=-1)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    0    30]
   [    1    11]
   [    1   101]
   [    1 10001]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([1 1 1 1 1 1], shape=(6,), dtype=int64),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))
  **Weighted bin-counting**
  This example takes two inputs - a values tensor and a weights tensor. These
  tensors must be identically shaped, and have the same row splits or indices
  in the case of RaggedTensors or SparseTensors. When performing a weighted
  count, the op will output a SparseTensor where the value of (i, j) is the
  sum of the values in the weight tensor's batch i in the locations where
  the values tensor has the value j. In this case, the output dtype is the
  same as the dtype of the weights tensor.
  >>> data = np.array([[10, 20, 30, 20], [11, 101, 11, 10001]], dtype=np.int64)
  >>> weights = [[2, 0.25, 15, 0.5], [2, 17, 3, 0.9]]
  >>> output = tf.sparse.bincount(data, weights=weights, axis=-1)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    0    30]
   [    1    11]
   [    1   101]
   [    1 10001]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([2. 0.75 15. 5. 17. 0.9], shape=(6,), dtype=float32),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))
  """
  with ops.name_scope(name, "count", [values, weights]):
    if not isinstance(values, sparse_tensor.SparseTensor):
      values = ragged_tensor.convert_to_tensor_or_ragged_tensor(
          values, name="values")
    if weights is not None:
      if not isinstance(weights, sparse_tensor.SparseTensor):
        weights = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            weights, name="weights")
    if weights is not None and binary_output:
      raise ValueError("Arguments `binary_output` and `weights` are mutually "
                       "exclusive. Please specify only one.")
    if axis is None:
      axis = 0
    if axis not in [0, -1]:
      raise ValueError(f"Unsupported value for argument axis={axis}. Only 0 and"
                       " -1 are currently supported.")
    minlength_value = minlength if minlength is not None else -1
    maxlength_value = maxlength if maxlength is not None else -1
    if axis == 0:
      if isinstance(values, sparse_tensor.SparseTensor):
        if weights is not None:
          weights = validate_sparse_weights(values, weights)
        values = values.values
      elif isinstance(values, ragged_tensor.RaggedTensor):
        if weights is not None:
          weights = validate_ragged_weights(values, weights)
        values = values.values
      else:
        if weights is not None:
          weights = array_ops.reshape(weights, [-1])
        values = array_ops.reshape(values, [-1])
    if isinstance(values, sparse_tensor.SparseTensor):
      weights = validate_sparse_weights(values, weights)
      c_ind, c_val, c_shape = gen_count_ops.sparse_count_sparse_output(
          values.indices,
          values.values,
          values.dense_shape,
          weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_output=binary_output)
    elif isinstance(values, ragged_tensor.RaggedTensor):
      weights = validate_ragged_weights(values, weights)
      c_ind, c_val, c_shape = gen_count_ops.ragged_count_sparse_output(
          values.row_splits,
          values.values,
          weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_output=binary_output)
    else:
      weights = validate_dense_weights(values, weights)
      c_ind, c_val, c_shape = gen_count_ops.dense_count_sparse_output(
          values,
          weights=weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_output=binary_output)
    return sparse_tensor.SparseTensor(c_ind, c_val, c_shape)
