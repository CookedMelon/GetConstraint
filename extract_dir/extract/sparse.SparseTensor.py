@tf_export("sparse.SparseTensor", "SparseTensor")
class SparseTensor(internal.NativeObject, composite_tensor.CompositeTensor):
  """Represents a sparse tensor.
  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.
  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:
  * `indices`: A 2-D int64 tensor of shape `[N, ndims]`, which specifies the
    indices of the elements in the sparse tensor that contain nonzero values
    (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]` specifies
    that the elements with indexes of [1,3] and [2,4] have nonzero values.
  * `values`: A 1-D tensor of any type and shape `[N]`, which supplies the
    values for each element in `indices`. For example, given `indices=[[1,3],
    [2,4]]`, the parameter `values=[18, 3.6]` specifies that element [1,3] of
    the sparse tensor has a value of 18, and element [2,4] of the tensor has a
    value of 3.6.
  * `dense_shape`: A 1-D int64 tensor of shape `[ndims]`, which specifies the
    dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.
  The corresponding dense tensor satisfies:
  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```
  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse.reorder(st)`.
  Example: The sparse tensor
  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```
  represents the dense tensor
  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  """
  @classmethod
  def from_value(cls, sparse_tensor_value):
    if not is_sparse(sparse_tensor_value):
      raise TypeError(f"Argument sparse_tensor_value={sparse_tensor_value} "
                      "is neither a SparseTensor nor SparseTensorValue.")
    return SparseTensor(
        indices=sparse_tensor_value.indices,
        values=sparse_tensor_value.values,
        dense_shape=sparse_tensor_value.dense_shape)
  def __init__(self, indices, values, dense_shape):
    """Creates a `SparseTensor`.
    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      dense_shape: A 1-D int64 tensor of shape `[ndims]`.
    Raises:
      ValueError: When building an eager SparseTensor if `dense_shape` is
        unknown or contains unknown elements (None or -1).
    """
    with ops.name_scope(None, "SparseTensor", [indices, values, dense_shape]):
      indices = ops.convert_to_tensor(
          indices, name="indices", dtype=dtypes.int64)
      # TODO(touts): Consider adding mutable_values() when 'values'
      # is a VariableOp and updating users of SparseTensor.
      values = ops.convert_to_tensor(values, name="values")
      dense_shape = ops.convert_to_tensor(
          dense_shape, name="dense_shape", dtype=dtypes.int64)
      dense_shape_default = tensor_util.constant_value_as_shape(dense_shape)
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape
    self._dense_shape_default = dense_shape_default
    indices_shape = indices.shape.with_rank(2)
    values_shape = values.shape.with_rank(1)
    dense_shape_shape = dense_shape.shape.with_rank(1)
    # Assert number of rows in indices match the number of elements in values.
    indices_shape.dims[0].assert_is_compatible_with(values_shape.dims[0])
    # Assert number of columns in indices matches the number of elements in
    # dense_shape.
    indices_shape.dims[1].assert_is_compatible_with(dense_shape_shape.dims[0])
  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.
    Returns:
      A `TensorShape` object.
    """
    return self._dense_shape_default
  @property
  def indices(self):
    """The indices of non-zero values in the represented dense tensor.
    Returns:
      A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the
        number of non-zero values in the tensor, and `ndims` is the rank.
    """
    return self._indices
  @property
  def values(self):
    """The non-zero values in the represented dense tensor.
    Returns:
      A 1-D Tensor of any data type.
    """
    return self._values
  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_values`.
    This method produces a new `SparseTensor` that has the same nonzero
    `indices` and same `dense_shape`, but updated values.
    Args:
      new_values: The values of the new `SparseTensor`. Needs to have the same
        shape as the current `.values` `Tensor`. May have a different type than
        the current `values`.
    Returns:
      A `SparseTensor` with identical indices and shape but updated values.
    Example usage:
    >>> st = tf.sparse.from_dense([[1, 0, 2, 0], [3, 0, 0, 4]])
    >>> tf.sparse.to_dense(st.with_values([10, 20, 30, 40]))  # 4 nonzero values
    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[10,  0, 20,  0],
           [30,  0,  0, 40]], dtype=int32)>
    """
    return SparseTensor(self._indices, new_values, self._dense_shape)
  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self._values.op
  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._values.dtype
  @property
  def dense_shape(self):
    """A 1-D Tensor of int64 representing the shape of the dense tensor."""
    return self._dense_shape
  @property
  def shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.
    Returns:
      A `TensorShape` object.
    """
    return self._dense_shape_default
  def set_shape(self, shape):
    """Updates the `TensorShape` representing the shape of the dense tensor.
    With eager execution this operates as a shape assertion.
    Here the shapes match:
    >>> st = tf.SparseTensor(
    ...   indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    >>> st.set_shape([3, 4])
    Passing a `None` in the new shape allows any value for that axis:
    >>> st.set_shape([3, None])
    An error is raised if an incompatible shape is passed.
    >>> st.set_shape([1, 4])
    Traceback (most recent call last):
    ...
    ValueError: Tensor's shape (3, 4) is not compatible with supplied
    shape [1, 4]
    When executing in a `tf.function`, or building a model using
    `tf.keras.Input`, `SparseTensor.set_shape` will *merge* the given `shape`
    with the current shape of this tensor, and set the tensor's shape to the
    merged value (see `tf.TensorShape.merge_with` for details):
    >>> st = tf.keras.Input(shape=[None, None, 3], sparse=True)
    >>> print(st.shape)
    (None, None, None, 3)
    Dimensions set to `None` are not updated:
    >>> st.set_shape([None, 224, 224, None])
    >>> print(st.shape)
    (None, 224, 224, 3)
    The main use case for this is to provide additional shape information
    that cannot be inferred from the graph alone.
    Caution: `set_shape` ensures that the applied shape is compatible with
    the existing shape, but it does not check at runtime. Setting
    incorrect shapes can result in inconsistencies between the
    statically-known graph and the runtime value of tensors.
    Args:
      shape: A `TensorShape` representing the shape of this tensor, a
        `TensorShapeProto`, a list, a tuple, or None.
    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    self._dense_shape_default = self._dense_shape_default.merge_with(shape)
  @property
  def graph(self):
    """The `Graph` that contains the index, value, and dense_shape tensors."""
    return self._indices.graph
  def __repr__(self):
    return "SparseTensor(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)
  def eval(self, feed_dict=None, session=None):
    """Evaluates this sparse tensor in a `Session`.
    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.
    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.
    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this sparse
        tensor. If none, the default session will be used.
    Returns:
      A `SparseTensorValue` object.
    """
    indices, values, dense_shape = _eval_using_default_session(
        [self.indices, self.values, self.dense_shape], feed_dict, self.graph,
        session)
    return SparseTensorValue(indices, values, dense_shape)
  @staticmethod
  def _override_operator(operator, func):
    _override_helper(SparseTensor, operator, func)
  @property
  def _type_spec(self):
    return SparseTensorSpec(self.shape, self.dtype)
  def _shape_invariant_to_type_spec(self, shape):
    # From the tf.while_loop docs: "If a loop variable is a SparseTensor, the
    # shape invariant must be TensorShape([r]) where r is the rank of the dense
    # tensor represented by the sparse tensor. It means the shapes of the three
    # tensors of the SparseTensor are ([None], [None, r], [r]). NOTE: The shape
    # invariant here is the shape of the SparseTensor.dense_shape property. It
    # must be the shape of a vector.
    if shape.ndims is not None and shape.ndims != 1:
      raise ValueError(f"Expected a shape with 1 dimension. Obtained: {shape} "
                       f"which has {shape.ndims} dimensions.")
    rank = tensor_shape.dimension_value(shape[0])
    return SparseTensorSpec(tensor_shape.unknown_shape(rank), self.dtype)
  def consumers(self):
    return self._consumers()
  def _numpy(self):
    """Returns a numpy `array` with the values for this `SparseTensor`.
    Requires that this `SparseTensor` was constructed in eager execution mode.
    """
    if not self._is_eager():
      raise ValueError("SparseTensor.numpy() is only supported in eager mode.")
    arr = np.zeros(self.dense_shape, dtype=self.dtype.as_numpy_dtype())
    for i, v in zip(self.indices, self.values):
      arr[tuple(i)] = v
    return arr
  def _is_eager(self):
    """Returns True if this `SparseTensor` was constructed in eager execution.
    Requires that each individual component of `SparseTensor`
    (`indices`, `values` and `dense_shape`) is an instance of `EagerTensor`.
    """
    return all(
        isinstance(t, ops.EagerTensor)
        for t in (self.indices, self.values, self.dense_shape))
