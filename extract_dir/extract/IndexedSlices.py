@tf_export("IndexedSlices")
class IndexedSlices(
    internal.IndexedSlices,
    internal.NativeObject,
    composite_tensor.CompositeTensor):
  """A sparse representation of a set of tensor slices at given indices.
  This class is a simple wrapper for a pair of `Tensor` objects:
  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.
  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.
  The dense tensor `dense` represented by an `IndexedSlices` `slices` has
  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```
  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).
  >>> v = tf.Variable([[0.,1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]])
  >>> with tf.GradientTape() as tape:
  ...   r = tf.gather(v, [1,3])
  >>> index_slices = tape.gradient(r,v)
  >>> index_slices
  <...IndexedSlices object ...>
  >>> index_slices.indices.numpy()
  array([1, 3], dtype=int32)
  >>> index_slices.values.numpy()
  array([[1., 1., 1.],
         [1., 1., 1.]], dtype=float32)
  Contrast this representation with
  `tf.sparse.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  """
  def __init__(self, values, indices, dense_shape=None):
    """Creates an `IndexedSlices`."""
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape
  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values
  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices
  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape
  @property
  def shape(self):
    """Gets the `tf.TensorShape` representing the shape of the dense tensor.
    Returns:
      A `tf.TensorShape` object.
    """
    if self._dense_shape is None:
      return tensor_shape.TensorShape(None)
    return tensor_util.constant_value_as_shape(self._dense_shape)
  @property
  def name(self):
    """The name of this `IndexedSlices`."""
    return self.values.name
  @property
  def device(self):
    """The name of the device on which `values` will be produced, or `None`."""
    return self.values.device
  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self.values.op
  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self.values.dtype
  @property
  def graph(self):
    """The `Graph` that contains the values, indices, and shape tensors."""
    return self._values.graph
  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values,
        (", dense_shape=%s" %
         (self._dense_shape,)) if self._dense_shape is not None else "")
  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)
  __composite_gradient__ = IndexedSlicesCompositeTensorGradient()
  @property
  def _type_spec(self):
    indices_shape = self._indices.shape.merge_with(self._values.shape[:1])
    dense_shape = tensor_shape.TensorShape([None]).concatenate(
        self._values.shape[1:])
    if self._dense_shape is not None:
      dense_shape_dtype = self._dense_shape.dtype
      dense_shape = dense_shape.merge_with(
          tensor_util.constant_value_as_shape(self._dense_shape))
    else:
      dense_shape_dtype = None
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)
  def _shape_invariant_to_type_spec(self, shape):
    # From tf.while_loop docs: "If a loop variable is an IndexedSlices, the
    # shape invariant must be a shape invariant of the values tensor of the
    # IndexedSlices. It means the shapes of the three tensors of the
    # IndexedSlices are (shape, [shape[0]], [shape.ndims])."
    indices_shape = shape[:1]
    dense_shape = tensor_shape.TensorShape([None]).concatenate(shape[1:])
    if self._dense_shape is None:
      dense_shape_dtype = None
    else:
      dense_shape_dtype = self._dense_shape.dtype
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)
  def consumers(self):
    return self._consumers()
