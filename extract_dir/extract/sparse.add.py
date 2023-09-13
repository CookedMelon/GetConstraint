@tf_export("sparse.add", v1=[])
def sparse_add_v2(a, b, threshold=0):
  """Adds two tensors, at least one of each is a `SparseTensor`.
  If one `SparseTensor` and one `Tensor` are passed in, returns a `Tensor`.  If
  both arguments are `SparseTensor`s, this returns a `SparseTensor`.  The order
  of arguments does not matter.  Use vanilla `tf.add()` for adding two dense
  `Tensor`s.
  The shapes of the two operands must match: broadcasting is not supported.
  The indices of any input `SparseTensor` are assumed ordered in standard
  lexicographic order.  If this is not the case, before this step run
  `SparseReorder` to restore index ordering.
  If both arguments are sparse, we perform "clipping" as follows.  By default,
  if two values sum to zero at some index, the output `SparseTensor` would still
  include that particular location in its index, storing a zero in the
  corresponding value slot.  To override this, callers can specify `threshold`,
  indicating that if the sum has a magnitude strictly smaller than `threshold`,
  its corresponding value and index would then not be included.  In particular,
  `threshold == 0.0` (default) means everything is kept and actual thresholding
  happens only for a positive value.
  For example, suppose the logical sum of two sparse operands is (densified):
      [       2]
      [.1     0]
      [ 6   -.2]
  Then,
  * `threshold == 0` (the default): all 5 index/value pairs will be
      returned.
  * `threshold == 0.11`: only .1 and 0 will vanish, and the remaining three
      index/value pairs will be returned.
  * `threshold == 0.21`: .1, 0, and -.2 will vanish.
  Args:
    a: The first operand; `SparseTensor` or `Tensor`.
    b: The second operand; `SparseTensor` or `Tensor`. At least one operand
      must be sparse.
    threshold: A 0-D `Tensor`. The magnitude threshold that determines if an
      output value/index pair takes space. Its dtype should match that of the
      values if they are real; if the latter are complex64/complex128, then the
      dtype should be float32/float64, correspondingly.
  Returns:
    A `SparseTensor` or a `Tensor`, representing the sum.
  Raises:
    TypeError: If both `a` and `b` are `Tensor`s.  Use `tf.add()` instead.
  """
  sparse_classes = (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)
  if not any(isinstance(inp, sparse_classes) for inp in [a, b]):
    raise TypeError("At least one input should be SparseTensor; do you mean to"
                    " use tf.add()?")
  if all(isinstance(inp, sparse_classes) for inp in [a, b]):
    a = _convert_to_sparse_tensor(a)
    b = _convert_to_sparse_tensor(b)
    threshold = ops.convert_to_tensor(
        threshold, dtype=a.values.dtype.real_dtype.base_dtype, name="threshold")
    output_ind, output_val, output_shape = (
        gen_sparse_ops.sparse_add(a.indices, a.values, a.dense_shape,
                                  b.indices, b.values, b.dense_shape,
                                  threshold))
    # Attempt to get output_shape statically.
    a.get_shape().assert_is_compatible_with(b.get_shape())
    static_shape = array_ops.broadcast_static_shape(a.get_shape(),
                                                    b.get_shape())
    if static_shape.is_fully_defined():
      output_shape = static_shape.as_list()
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
  else:
    # swap to make `a` the SparseTensor.
    if isinstance(b, sparse_classes):
      a, b = b, a
    return gen_sparse_ops.sparse_tensor_dense_add(a.indices, a.values,
                                                  a.dense_shape, b)
