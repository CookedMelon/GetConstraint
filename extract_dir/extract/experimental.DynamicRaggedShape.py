@tf_export("experimental.DynamicRaggedShape")
class DynamicRaggedShape(extension_type.BatchableExtensionType):
  """The shape of a ragged or dense tensor.
  Ragged shapes are encoded using two fields:
  * `inner_shape`: An integer vector giving the shape of a dense tensor.
  * `row_partitions`: A list of `RowPartition` objects, describing how
    that flat shape should be partitioned to add ragged axes.
  If a DynamicRaggedShape is the shape of a RaggedTensor rt, then:
  1. row_partitions = rt._nested_row_partitions
     (and thus len(row_partitions) > 0)
  2. inner_shape is the shape of rt.flat_values
  If a DynamicRaggedShape is the shape of a dense tensor t, then:
  1. row_partitions = []
  2. inner_shape is the shape of t.
  Examples:
  The following table gives a few examples (where `RP(lengths)` is short
  for `RowPartition.from_lengths(lengths)`):
  Row Partitions              | Inner Shape  | Example Tensor
  --------------------------- | ------------ | ----------------------------
  []                          | [2, 3]       | `[[1, 2, 3], [4, 5, 6]]`
  [RP([2, 0, 3])]             | [5]          | `[[1, 2], [], [3, 4, 5]]`
  [RP([2, 1])]                | [3, 2]       | `[[[1, 2], [3, 4]], [[5, 6]]]`
  [RP([2, 1]), RP([2, 1, 2])] | [5]          | `[[[1, 2], [3]], [[4, 5]]]`
  """
  _row_partitions: Tuple[RowPartition, ...]
  _inner_shape: ops.Tensor
  _static_inner_shape: tensor_shape.TensorShape
  __batch_encoder__ = _DynamicRaggedShapeBatchEncoder()
  __name__ = "tf.DynamicRaggedShape"
  def __init__(self,
               row_partitions: Sequence[RowPartition],
               inner_shape: core.TensorLike,
               dtype: Optional[dtypes.DType] = None,
               validate: bool = False,
               static_inner_shape: ... = None):
    """Core constructor for a DynamicRaggedShape.
    Create a DynamicRaggedShape. This can be used to construct a
    DynamicRaggedShape representing a ragged or dense shape. If row_partitions
    is an empty list, then this is equivalent to a dense shape.
    If row_partitions is specified, then the num_row_partitions will be equal
    to len(row_partitions). There are several checks made.
    Specifically:
    1. Consecutive row_partitions must have consistent nvals and nrows.
    2. The last row_partitions must have nvals equal to the first element of
       inner_shape.
    The inner_shape is converted to a tensor.
    All row_partitions and the inner_shape are converted to the same dtype
    (int64 or int32).
    Args:
      row_partitions: the row_partitions of the shape.
      inner_shape: if len(row_partitions) > 0, the shape of the flat_values.
        Otherwise, the shape of the tensor.
      dtype: tf.int64, tf.int32, or None representing the preferred dtype.
      validate: if true, dynamic validation is applied to the shape.
      static_inner_shape: if len(row_partitions) > 0, the static shape of the
        flat_values. Otherwise, the static shape of the tensor. Should be
        convertible to a TensorShape.
    """
    if not isinstance(row_partitions, Iterable):
      raise TypeError(
          "row_partitions should be a list of row partitions. Instead, got " +
          str(row_partitions))
    for x in row_partitions:
      if not isinstance(x, RowPartition):
        raise TypeError("row_partitions contains " + str(x) +
                        " which is not a RowPartition")
    dtype = _find_dtype_iterable(row_partitions, dtype)
    dtype = _find_dtype(inner_shape, dtype)
    if (isinstance(inner_shape, np.ndarray) and
        inner_shape.dtype == np.int32 and dtype is None):
      dtype = dtypes.int32
    dtype = _find_dtype(dtypes.int64, dtype)
    row_partitions = tuple([rp.with_dtype(dtype) for rp in row_partitions])
    self._row_partitions = row_partitions
    self._inner_shape = ops.convert_to_tensor(
        inner_shape, dtype_hint=dtype, name="inner_dim_sizes")
    if self._inner_shape.dtype != dtype:
      self._inner_shape = math_ops.cast(self._inner_shape, dtype)
    checks = []
    # Validate shapes.
    if self._row_partitions:
      for axis, rp in enumerate(self._row_partitions):
        if axis > 0:
          previous_row_partition = self._row_partitions[axis - 1]
          msg = ("RowPartitions in DynamicRaggedShape do not align "
                 f"between {axis - 1} and {axis}")
          static_nrows = rp.static_nrows
          static_nvals = previous_row_partition.static_nvals
          if (static_nrows is not None) and (static_nvals is not None):
            if static_nrows != static_nvals:
              raise ValueError(msg)
            else:
              continue
          if validate:
            checks.append(
                check_ops.assert_equal(
                    previous_row_partition.nvals(), rp.nrows(), message=msg))
    self._inner_shape.shape.assert_has_rank(1)
    self._static_inner_shape = tensor_util.constant_value_as_shape(
        self._inner_shape)
    if static_inner_shape is not None:
      self._static_inner_shape = self._static_inner_shape.merge_with(
          static_inner_shape)
    if row_partitions:
      last_row_partition = row_partitions[-1]
      static_nvals = last_row_partition.static_nvals
      static_inner_shape_nvals = tensor_shape.dimension_value(
          self._static_inner_shape[0])
      if static_nvals is not None and static_inner_shape_nvals is not None:
        if static_nvals != static_inner_shape_nvals:
          raise ValueError("Last row partition does not match inner_shape.")
      elif validate:
        checks.append(
            check_ops.assert_equal(
                last_row_partition.nvals(),
                self._inner_shape[0],
                message="Last row partition does not match inner_shape."))
    if checks:
      self._inner_shape = control_flow_ops.with_dependencies(
          checks, self._inner_shape, name="inner_shape_validated")
      self._row_partitions = [
          rp._with_dependencies(checks) for rp in self._row_partitions  # pylint: disable=protected-access
      ]
  @classmethod
  def from_lengths(cls,
                   lengths: Sequence[Union[Sequence[int], int]],
                   num_row_partitions=None,
                   dtype=dtypes.int64):
    """Creates a shape with the given lengths and num_row_partitions.
    The lengths can either be a nonnegative int or a list of nonnegative ints.
    If num_row_partitions is None, then the minimal num_row_partitions is used.
    For example, [2, (3, 2)] is the shape of [[0, 0, 0], [0, 0]], and
    [2, 2] is the shape of [[0, 0], [0, 0]]
    This chooses the minimal num_row_partitions required (including zero).
    The following table gives a few examples (where `RP(lengths)` is short
    for `RowPartition.from_lengths(lengths)`):
    For example:
    from_lengths           | row_partitions            | inner_shape
    ---------------------- | --------------------------| -------------
    []                     | []                        | []
    [2, (3, 2)]            | [RP([3, 2])]              | [5]
    [2, 2]                 | []                        | [2, 2]
    [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
    [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
    [2, 2, 3]              | []                        | [2, 2, 3]
    [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]
    If we want the row partitions to end with uniform row partitions, then
    we can set num_row_partitions.
    For example,
    below URP(3, 12) is RowPartition.from_uniform_row_length(3, 12)
    from_lengths   | num_row_partitions | row_partitions           | inner_shape
    ---------------| -------------------|--------------------------|------------
    [2, (3, 2), 2] | 2                  | [RP([3, 2]), URP(2, 10)] | [10]
    [2, 2]         | 1                  | [URP(2, 4)]              | [4]
    [2, 2, 3]      | 0                  | []                       | [2, 2, 3]
    [2, 2, 3]      | 1                  | [URP(2, 4)]              | [4, 3]
    [2, 2, 3]      | 2                  | [URP(2, 4), URP(3, 12)]  | [12]
    Representing the shapes from init():
    from_lengths             | Tensor Example
    ------------------------ | ------------------------------
    `[2, 3]`                 | `[[1, 2, 3], [4, 5, 6]]`
    `[3, (2, 0, 3)]`         | `[[1, 2], [], [3, 4, 5]]`
    `[2, (2, 1), 2]`         | `[[[1, 2], [3, 4]], [[5, 6]]]`
    `[2, (2, 1), (2, 1, 2)]` | `[[[1, 2], [3]], [[4, 5]]]`
    Args:
      lengths: the lengths of sublists along each axis.
      num_row_partitions: the num_row_partitions of the result or None
        indicating the minimum number of row_partitions.
      dtype: the dtype of the shape (tf.int32 or tf.int64).
    Returns:
      a new DynamicRaggedShape
    """
    if not isinstance(lengths, list):
      raise ValueError("lengths should be a list")
    for x in lengths:
      if not _is_int_or_tuple_of_ints(x):
        raise ValueError(
            "element of lengths should be int or tuple of ints: instead %r" %
            (x,))
    if num_row_partitions is None:
      # Calculate the minimal num_row_partitions.
      is_list = [not isinstance(x, int) for x in lengths]
      if any(is_list):
        # Last index when not a list.
        num_row_partitions = len(is_list) - is_list[-1::-1].index(True) - 1
      else:
        num_row_partitions = 0
    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions should be an int or None")
    if not lengths:
      if num_row_partitions > 0:
        raise ValueError("num_row_partitions==0 for a scalar shape")
      return DynamicRaggedShape([], [], dtype=dtype)
    if not num_row_partitions < len(lengths):
      raise ValueError("num_row_partitions should be less than `len(lengths)` "
                       "if shape is not scalar.")
    if num_row_partitions > 0:
      (row_partitions, nvals) = _to_row_partitions_and_nvals_from_lengths(
          lengths[:num_row_partitions + 1])
      inner_shape = [nvals] + lengths[num_row_partitions + 1:]
      return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)
    else:
      return DynamicRaggedShape([], lengths, dtype=dtype)
  @classmethod
  def from_row_partitions(cls, row_partitions, dtype=None):
    """Create a shape from row_partitions.
    Args:
      row_partitions: a nonempty list of RowPartition objects.
      dtype: the dtype to use, or None to use the row_partitions dtype.
    Returns:
      a DynamicRaggedShape with inner_rank==1.
    """
    if not row_partitions:
      raise ValueError("row_partitions cannot be empty")
    inner_shape = [row_partitions[-1].nvals()]
    return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)
  @classmethod
  def _from_inner_shape(cls, inner_shape, dtype=None):
    """Create a shape from inner_shape, where num_row_partitions == 0."""
    return DynamicRaggedShape([], inner_shape, dtype=dtype)
  # pylint: disable=protected-access
  @classmethod
  def from_tensor(cls, t, dtype=None):
    """Constructs a ragged shape for a potentially ragged tensor."""
    if ragged_tensor.is_ragged(t):
      return DynamicRaggedShape(
          t._nested_row_partitions, _flat_values_shape(t), dtype=dtype)
    else:
      return DynamicRaggedShape._from_inner_shape(
          array_ops.shape(t), dtype=dtype)
  @property
  def row_partitions(self):
    """The row_partitions of the shape."""
    return self._row_partitions
  @property
  def num_row_partitions(self):
    """The number of row_partitions of the shape."""
    return len(self._row_partitions)
  @property
  def dtype(self):
    """The dtype of the shape -- one of tf.int32 or tf.int64."""
    return self._inner_shape.dtype
  def _static_inner_shape_as_list(self, truncate_first):
    """Returns the lengths of the inner shape (if rank known), or [...]."""
    if self._static_inner_shape.rank is None:
      return [...]
    result = self._static_inner_shape.as_list()
    if truncate_first:
      return result[1:]
    return result
  def static_lengths(self, ragged_lengths=True):
    """Returns a list of statically known axis lengths.
    This represents what values are known. For each row partition, it presents
    either the uniform row length (if statically known),
    the list of row lengths, or none if it is not statically known.
    For the inner shape, if the rank is known, then each dimension is reported
    if known, and None otherwise. If the rank of the inner shape is not known,
    then the returned list ends with an ellipsis.
    Args:
      ragged_lengths: If false, returns None for all ragged dimensions.
    Returns:
      A Sequence[Union[Sequence[int],int, None]] of lengths, with a possible
      Ellipsis at the end.
    """
    if self.num_row_partitions == 0:
      return self._static_inner_shape_as_list(False)
    first_dim = self.row_partitions[0].static_nrows
    if isinstance(first_dim, tensor_shape.Dimension):
      first_dim = first_dim.value
    rp_dims = [first_dim]
    for rp in self.row_partitions:
      if rp.is_uniform():
        rp_dims.append(rp.static_uniform_row_length)
      elif ragged_lengths:
        const_vals = tensor_util.constant_value(rp.row_lengths())
        if const_vals is None:
          rp_dims.append(None)
        else:
          rp_dims.append(tuple(const_vals.tolist()))
      else:
        rp_dims.append(None)
    return rp_dims + self._static_inner_shape_as_list(True)
  def __repr__(self):
    lengths = _list_with_ellipsis_to_str(self.static_lengths())
    return ("<DynamicRaggedShape "
            "lengths=%s num_row_partitions=%r>" %
            (lengths, self.num_row_partitions))
  def _to_tensor_shape(self) -> tensor_shape.TensorShape:
    """Returns a TensorShape representation of the shape."""
    lengths = self.static_lengths(ragged_lengths=False)
    if not lengths:
      return tensor_shape.TensorShape(())
    if lengths[-1] == Ellipsis:
      return tensor_shape.TensorShape(None)
    return tensor_shape.TensorShape(lengths)
  def _slice_shape(self, start, stop):
    """Returns a shape self[start:stop].
    If start == 0, then this truncates dimensions after stop.
    If start != 0, then this will return a shape with num_row_partitions == 0.
    See __getitem__.
    Args:
      start: the first dimension. 0 <= start <= rank
      stop: the last dimension (exclusive). 0 <= stop <= rank
    """
    if stop <= start:
      return DynamicRaggedShape._from_inner_shape([])
    elif start == 0:
      if stop <= self.num_row_partitions:
        if stop == 1:
          return DynamicRaggedShape._from_inner_shape(
              [self.row_partitions[0].nrows()])
        new_row_partitions = self.row_partitions[:stop - 1]
        new_inner_shape = [new_row_partitions[-1].nvals()]
        return DynamicRaggedShape(new_row_partitions, new_inner_shape)
      else:
        if self.rank is None:
          new_inner_rank = stop - self.num_row_partitions
          new_inner_shape = self.inner_shape[:new_inner_rank]
          return DynamicRaggedShape(
              row_partitions=self.row_partitions,
              inner_shape=new_inner_shape,
              static_inner_shape=None,
              validate=False)
        elif self.rank <= stop:
          return self
        new_inner_rank = stop - self.num_row_partitions
        new_inner_shape = self.inner_shape[:new_inner_rank]
        return DynamicRaggedShape(
            row_partitions=self.row_partitions,
            inner_shape=new_inner_shape,
            static_inner_shape=tensor_shape.TensorShape([None] *
                                                        new_inner_rank),
            validate=False)
    else:
      if self.rank is None or stop < self.rank:
        partial = self._slice_shape(0, stop)
      else:
        partial = self
      for x in partial.row_partitions:
        if not x.is_uniform():
          raise ValueError("All relevant dimensions must be uniform")
      if partial.rank is None:
        # TODO(martinz): Implement _with_num_row_partitions(0) if rank is
        # unknown, and remove.
        raise NotImplementedError(
            "__getitem__[start:stop] where start > 0 not implemented")
      return DynamicRaggedShape._from_inner_shape(
          partial._with_num_row_partitions(0).inner_shape[start:])
  def _dimension(self, index):
    """Return a dimension, if the dimension is not ragged (see __getitem__)."""
    rank = self.rank
    if not isinstance(index, int):
      raise TypeError("index should be an int")
    if (self.num_row_partitions == 0 or index > self.num_row_partitions + 1):
      # If num_row_partitions > 0 and index <= num_row_partitions + 1, then
      # we are safe.
      if rank is None:
        raise ValueError(
            "Rank must be known to use __getitem__ on a large index.")
      if index >= rank:
        raise IndexError("Index is too big: " + str(index) + ">=" + str(rank))
    if index < 0:
      raise IndexError("Index must be non-negative: " + str(index))
    elif not self.is_uniform(index):
      raise ValueError("Index " + str(index) + " is not uniform")
    elif index == 0 and self.num_row_partitions > 0:
      static_nrows = self.row_partitions[0].static_nrows
      if static_nrows is not None:
        return constant_op.constant(static_nrows, dtype=self.dtype)
      return self.row_partitions[0].nrows()
    elif self.num_row_partitions == 0:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)
      return self.inner_shape[index]
    elif index > self.num_row_partitions:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index - self.num_row_partitions])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)
      return self.inner_shape[index - self.num_row_partitions]
    else:
      return self.row_partitions[index - 1].uniform_row_length()
  def __getitem__(self, index):
    """Returns a dimension or a slice of the shape.
    Ragged shapes can have ragged dimensions that depend upon other dimensions.
    Therefore, if you ask for a dimension that is ragged, this function returns
    a ValueError. For similar reasons, if a slice is selected that includes
    a ragged dimension without including the zero dimension, then this fails.
    Any slice that does not start at zero will return a shape
    with num_row_partitions == 0.
    Args:
      index: the index: can be an int or a slice.
    Raises:
      IndexError: if the index is not in range.
      ValueError: if the rank is unknown, or a ragged rank is requested
      incorrectly.
    """
    rank = self.rank
    if isinstance(index, slice):
      if (index.step is not None) and (index.step != 1):
        raise IndexError("Cannot stride through a shape")
      start = index.start
      stop = index.stop
      if start is None:
        start = 0
      start = _fix_start_index(start, rank, self.num_row_partitions)
      stop = _fix_stop_index(stop, rank)
      return self._slice_shape(start, stop)
    elif isinstance(index, int):
      if index < 0:
        if rank is None:
          raise ValueError(
              "Rank must be known to use __getitem__ with a negative index.")
        return self._dimension(rank + index)
      return self._dimension(index)
    else:
      raise TypeError("Argument is not an int or a slice")
  def _num_elements(self):
    """Number of elements in a shape.
    Returns:
      The number of elements in the shape.
    """
    return math_ops.reduce_prod(self.inner_shape)
  def _num_slices_in_dimension(self, axis):
    """The total size of a dimension (like nvals).
    Effectively, this is self[:axis+1]._num_elements()
    Example:
    shape = DynamicRaggedShape._from_inner_shape([2, 3, 4])
    shape._num_slices_in_dimension(0) = 2
    shape._num_slices_in_dimension(1) = 6
    shape._num_slices_in_dimension(2) = 24
    shape._num_slices_in_dimension(-1) = 24
    shape._num_slices_in_dimension(-2) = 6
    shape._num_slices_in_dimension(-2) = 2
    Args:
      axis: the last axis to include in the number of elements. If negative,
        then axis = axis + rank.
    Returns:
      The number of elements in the shape.
    """
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    if axis < 0:
      rank = self.rank
      if rank is None:
        raise ValueError(
            "You can't use negative values if the rank is undefined")
      axis = axis + rank
    if axis == 0:
      return self._dimension(0)
    if axis <= self.num_row_partitions:
      return self.row_partitions[axis - 1].nvals()
    # If self.num_row_partitions = 1, and
    # self.inner_shape=[3,5,6], and axis=2, then you want:
    # 15 = 3 * 5 = math_ops.reduce_prod(self.inner_shape[:2])
    # 2 = axis - (self.num_row_partitions - 1)
    # If num_row_partitions=0, and
    # self.inner_shape=[3,5,6] and axis=2, then you want:
    # 90 = 3 * 5 * 6 = math_ops.reduce_prod(self.inner_shape[:3])
    # 3 = axis - (self.num_row_partitions - 1)
    remainder = axis - (self.num_row_partitions - 1)
    return _reduce_prod_patch(self.inner_shape[:remainder])
  def is_uniform(self, axis):
    """Returns true if the indicated dimension is uniform."""
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    rank = self.rank
    if axis < 0:
      raise IndexError("Negative axis values are not supported")
    elif rank is not None and axis >= rank:
      raise IndexError("Expected axis=%s < rank=%s" % (axis, rank))
    else:
      return ((axis == 0 or axis > len(self._row_partitions))  # pylint:disable=superfluous-parens
              or self._row_partitions[axis - 1].is_uniform())
  @property
  def rank(self):
    """The number of dimensions in this shape, or None if unknown."""
    inner_rank = self.inner_rank
    if inner_rank is None:
      return None
    else:
      return self.num_row_partitions + inner_rank
  @property
  def inner_shape(self):
    """The inner dimension sizes for this shape.
    Returns:
      A 1-D integer `Tensor`.
    """
    return self._inner_shape
  @property
  def inner_rank(self):
    """The rank of inner_shape."""
    return tensor_shape.dimension_value(self._static_inner_shape.rank)
  def _alt_inner_shape(self, new_inner_rank):
    """Get an alternative inner shape with higher or lower rank.
    For the rank of the inner shape to be be higher, the last few ragged
    dimensions must have uniform_row_length.
    Args:
      new_inner_rank: the new rank of the inner_shape
    Returns:
       A new inner_shape of rank new_inner_rank.
    """
    if new_inner_rank == 0:
      raise ValueError("new_inner_rank cannot be zero")
    elif self.inner_rank == 0:
      raise ValueError("old inner_rank cannot be zero")
    elif new_inner_rank == self.inner_rank:
      return self.inner_shape
    elif new_inner_rank < self.inner_rank:
      if self._static_inner_shape.is_fully_defined():
        return _alt_inner_shape_from_tensor_shape(self._static_inner_shape,
                                                  self.dtype, new_inner_rank)
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      if new_inner_rank == 1:
        return array_ops.expand_dims(first_dimension, 0)
      remaining_dimensions = self.inner_shape[1 - new_inner_rank:]
      return array_ops.concat(
          [array_ops.expand_dims(first_dimension, 0), remaining_dimensions],
          axis=0)
    else:
      assert new_inner_rank > self.inner_rank
      new_dimensions = new_inner_rank - self.inner_rank
      if any(
          [not x.is_uniform() for x in self.row_partitions[-new_dimensions:]]):
        raise ValueError("Cannot get an inner shape over a ragged dimension")
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      new_dimensions = new_inner_rank - self.inner_rank
      new_dims = [first_dimension] + [
          x.uniform_row_length() for x in self.row_partitions[-new_dimensions:]
      ]
      return array_ops.concat(
          [array_ops_stack.stack(new_dims), self.inner_shape[1:]], axis=0)
  def _inner_shape_dim(self, dimension):
    """Returns an int or a tensor representing _inner_shape[dimension]."""
    result = tensor_shape.dimension_value(self._static_inner_shape[dimension])
    return self._inner_shape[dimension] if result is None else result
  def _with_inner_rank(self, inner_rank):
    """Returns the same shape but a different inner_rank.
    All dimensions that are to be represented in the inner_shape must be dense.
    See inner_rank.
    Args:
      inner_rank: the new inner_rank of the shape.
    Returns:
      the same shape but a different inner_rank
    Raises:
      ValueError if the new dense rank is invalid, or the old rank is unknown.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust inner_rank")
    elif rank < 2:
      if inner_rank == rank:
        return self
      raise ValueError("Cannot change inner_rank if rank < 2")
    else:
      # When self.rank is not None:
      # self.rank = self.inner_rank + self.num_row_partitions
      new_num_row_partitions = rank - inner_rank
      return self._with_num_row_partitions(new_num_row_partitions)
  def _with_num_row_partitions(self, num_row_partitions):
    """Creates an identical shape with the given num_row_partitions.
    Note that the shape must be statically refactorable to this rank.
    In particular:
    * rank must be known.
    * num_row_partitions must be a nonnegative int.
    * num_row_partitions must be less than the rank of the shape
    * num_row_partitions must be greater or equal to the index of any ragged
    dimension.
    Note that if the num_row_partitions is the same, self is returned.
    Args:
      num_row_partitions: the target num_row_partitions (must be a nonnegative
        int).
    Returns:
      a shape with a (possibly) different num_row_partitions.
    Raises:
      ValueError: if the rank is unknown, the argument is not a nonnegative int,
        or there is a dimension that is nonuniform.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust num_row_partitions")
    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions must be an int")
    if num_row_partitions < 0:
      raise ValueError("num_row_partitions must be nonnegative")
    if num_row_partitions == self.num_row_partitions:
      return self
    if num_row_partitions >= rank:
      raise ValueError("num_row_partitions must be less than rank")
    if num_row_partitions > self.num_row_partitions:
      num_row_partitions_diff = num_row_partitions - self.num_row_partitions
      new_inner_rank = self.rank - num_row_partitions
      nvals = self._inner_shape_dim(0)
      more_rp = []
      for i in range(num_row_partitions_diff):
        nrows = nvals
        row_length = self._inner_shape_dim(i + 1)
        nvals = nrows * row_length
        rp = RowPartition.from_uniform_row_length(
            row_length, nrows=nrows, dtype=self.dtype)
        more_rp.append(rp)
      alt_inner = self._alt_inner_shape(new_inner_rank)
      return DynamicRaggedShape(list(self.row_partitions) + more_rp, alt_inner)
    else:
      assert num_row_partitions < self.num_row_partitions
      return DynamicRaggedShape(
          self.row_partitions[:num_row_partitions],
          self._alt_inner_shape(self.rank - num_row_partitions))
  def _merge_dims(self, outer_axis: int,
                  inner_axis: int) -> "DynamicRaggedShape":
    """Merges outer_axis...inner_axis into a single dimension.
    Returns a copy of this shape with the specified range of dimensions
    flattened into a single dimension, with elements in row-major order.
    #### Examples:
    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(0, 1)
    <DynamicRaggedShape lengths=[3, (1, 2, 3)] num_row_partitions=1>
    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(1, 2)
    <DynamicRaggedShape lengths=[2, (3, 3)] num_row_partitions=1>
    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(0, 2)
    <DynamicRaggedShape lengths=[6] num_row_partitions=0>
    To mimic the behavior of `np.flatten` (which flattens all dimensions), use
    `rt.merge_dims(0, -1).  To mimic the behavior of `tf.layers.Flatten` (which
    flattens all dimensions except the outermost batch dimension), use
    `rt.merge_dims(1, -1)`.
    Args:
      outer_axis: `int`: The first dimension in the range of dimensions to
        merge. May be negative if `self.shape.rank` is statically known.
      inner_axis: `int`: The last dimension in the range of dimensions to merge.
        May be negative if `self.shape.rank` is statically known.
    Returns:
      A copy of this shape, with the specified dimensions merged into a
      single dimension.  The returned shape will be
      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`
      is the total number of slices in the merged dimensions.
    """
    outer_axis = array_ops.get_positive_axis(
        outer_axis, self.rank, axis_name="outer_axis", ndims_name="rank(self)")
    inner_axis = array_ops.get_positive_axis(
        inner_axis, self.rank, axis_name="inner_axis", ndims_name="rank(self)")
    if not outer_axis <= inner_axis:
      raise ValueError(f"Expected outer_axis ({outer_axis}) to be less than or "
                       f"equal to inner_axis ({inner_axis}).")
    if outer_axis == inner_axis:
      return self
    if self.num_row_partitions == 0:
      # A dense tensor.
      (new_inner_shape,
       new_static_inner_shape) = _merge_inner_shape(self._inner_shape,
                                                    self._static_inner_shape,
                                                    outer_axis, inner_axis)
      return DynamicRaggedShape([],
                                new_inner_shape,
                                dtype=self.dtype,
                                static_inner_shape=new_static_inner_shape)
    if inner_axis <= self.num_row_partitions:
      # Here, we are merging the row_partitions,
      # but the inner_shape is unchanged.
      if outer_axis == 0:
        # There is no need to merge axes before the first, just truncate them.
        return DynamicRaggedShape(
            self._row_partitions[inner_axis:],
            self.inner_shape,
            dtype=self.dtype,
            static_inner_shape=self._static_inner_shape)
      prefix_rp = self._row_partitions[:outer_axis - 1]
      suffix_rp = self._row_partitions[inner_axis:]
      internal_rp = self._row_partitions[outer_axis - 1:inner_axis]
      new_rp = prefix_rp + (_merge_row_partitions(internal_rp),) + suffix_rp
      return DynamicRaggedShape(
          new_rp,
          self.inner_shape,
          dtype=self.dtype,
          static_inner_shape=self._static_inner_shape)
    elif outer_axis > self.num_row_partitions:
      # In this scenario, only the inner_shape is changed.
      # Example #1:
      # if [2, (1, 2), 5, 3], num_row_partitions=1, outer_axis=2, inner_axis=3.
      # Result: [2, (1, 2), 15], num_row_partitions=1, outer_axis=2,
      #     inner_axis=3.
      (new_inner_shape, new_static_inner_shape) = _merge_inner_shape(
          self._inner_shape, self._static_inner_shape,
          outer_axis - self.num_row_partitions,
          inner_axis - self.num_row_partitions)
      return DynamicRaggedShape(
          self._row_partitions,
          new_inner_shape,
          dtype=self.dtype,
          static_inner_shape=new_static_inner_shape)
    else:
      # Here, both inner_shape and row_partitions are changed.
      rank = self.rank
      if rank is None:
        raise ValueError("Cannot merge_dims of the inner shape if the " +
                         "dimension of inner_shape is unknown")
      if outer_axis == 0:
        new_inner_shape = self._alt_inner_shape(rank - inner_axis)
        return DynamicRaggedShape._from_inner_shape(new_inner_shape)
      else:
        prefix = self._row_partitions[:outer_axis - 1]
        suffix = _merge_row_partitions(self._row_partitions[outer_axis - 1:])
        new_inner_shape = self._alt_inner_shape(rank - inner_axis)
        num_merged_inner = inner_axis - self.num_row_partitions
        prod = _reduce_prod_patch(self._inner_shape[1:num_merged_inner + 1])
        tail_suffix = RowPartition.from_row_splits(suffix.row_splits() * prod)
        return DynamicRaggedShape(prefix + (tail_suffix,), new_inner_shape)
  def with_dtype(self, dtype):
    """Change the dtype of the shape."""
    if dtype == self.dtype:
      return self
    else:
      return DynamicRaggedShape(
          self.row_partitions, self.inner_shape, dtype=dtype)
  def _merge_with(self, other: "DynamicRaggedShape") -> "DynamicRaggedShape":
    """Merge two shapes that are equal modulo num_row_partitions.
    The resulting num_row_partitions is the maximum of the two
    num_row_partitions.
    Args:
      other: a DynamicRaggedShape representing the same shape with a possibly
        different number of row partitions.
    Returns:
      A DynamicRaggedShape with the same shape and the maximum of the
      num_row_partitions of the two shapes.
    """
    max_num_row_partitions = max(self.num_row_partitions,
                                 other.num_row_partitions)
    a = self._with_num_row_partitions(max_num_row_partitions)
    b = other._with_num_row_partitions(max_num_row_partitions)
    new_row_partitions = [
        rp_a._merge_precomputed_encodings(rp_b)
        for (rp_a, rp_b) in zip(a._row_partitions, b._row_partitions)
    ]
    new_dtype = b.dtype if a.dtype == dtypes.int32 else dtypes.int64
    new_static_inner_shape = a._static_inner_shape.merge_with(
        b._static_inner_shape)
    new_inner_shape = a._inner_shape
    return DynamicRaggedShape(new_row_partitions, new_inner_shape, new_dtype,
                              True, new_static_inner_shape)
  def _merge_with_spec(
      self, other: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape":
    """Merge a spec with a DynamicRaggedShape."""
    # TODO(martinz): add tests for dynamic inconsistencies.
    max_num_row_partitions = max(self.num_row_partitions,
                                 other.num_row_partitions)
    a = self._with_num_row_partitions(max_num_row_partitions)
    b = other._with_num_row_partitions(max_num_row_partitions)
    new_row_partitions = [
        rp_a._merge_with_spec(rp_b)
        for (rp_a, rp_b) in zip(a._row_partitions, b._row_partitions)
    ]
    new_dtype = b.dtype if a.dtype == dtypes.int32 else dtypes.int64
    new_static_inner_shape = a._static_inner_shape.merge_with(
        b._static_inner_shape)
    new_inner_shape = a._inner_shape
    return DynamicRaggedShape(new_row_partitions, new_inner_shape, new_dtype,
                              True, new_static_inner_shape)
  def _as_row_partitions(self):
    """Returns row partitions representing this shape.
    In order to represent a shape as row partitions, the rank of the shape
    must be known, and the shape must have rank at least one.
    Returns:
      A list of RowPartition objects.
    Raises:
      ValueError, if the shape cannot be represented by RowPartitions.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("rank must be known for _as_row_partitions")
    elif rank < 1:
      raise ValueError("rank must be >= 1 for _as_row_partitions")
    fully_ragged = self._with_num_row_partitions(rank - 1)
    return fully_ragged.row_partitions
  def _validate_flat_values_dynamically(self, flat_values):
    """Test if flat_values have the right nvals dynamically."""
    if self.row_partitions:
      assert_op = check_ops.assert_equal(
          self.row_partitions[-1].nvals(),
          array_ops.shape(flat_values, out_type=self.dtype)[0],
          message="Last row partition does not match flat_values.")
      return control_flow_ops.with_dependencies([assert_op], flat_values)
    return flat_values
  def _validate_flat_values(self, flat_values):
    """Test if flat_values have the right nvals."""
    if not isinstance(flat_values, ops.Tensor):
      return flat_values
    if self.row_partitions:
      last_row_partition = self.row_partitions[-1]
      flat_values_shape = flat_values.shape
      if flat_values_shape is None:
        return self._validate_flat_values_dynamically(flat_values)
      first_dim_flat_values = flat_values_shape[0]
      if isinstance(first_dim_flat_values, tensor_shape.Dimension):
        first_dim_flat_values = first_dim_flat_values.value
      if first_dim_flat_values is None:
        return self._validate_flat_values_dynamically(flat_values)
      static_nvals = last_row_partition.static_nvals
      if static_nvals is None:
        return self._validate_flat_values_dynamically(flat_values)
      if first_dim_flat_values != static_nvals:
        raise ValueError("Last row partition does not match flat_values.")
    return flat_values
  def _add_row_partitions(self, flat_values, validate=False):
    """Add row partitions to flat_values, if necessary.
    If the shape is truly ragged, then this adds the row_partitions.
    The shape is dense, then this just returns flat_values.
    Args:
      flat_values: the flat_values of a ragged tensor with this shape, or a
        dense tensor with this shape.
      validate: validate the flat_values have the right first dimension.
    Returns:
      flat_values reshaped to have row_partitions.
    """
    if self.row_partitions:
      if validate:
        flat_values = self._validate_flat_values(flat_values)
      return ragged_tensor.RaggedTensor._from_nested_row_partitions(
          flat_values, self.row_partitions, validate=False)
    else:
      return flat_values
  class Spec:
    """A Spec for DynamicRaggedShape: similar to a static shape."""
    def __init__(self, row_partitions: Tuple[RowPartitionSpec, ...],
                 static_inner_shape: tensor_shape.TensorShape,
                 dtype: dtypes.DType):
      """Create a Spec given row partitions, a static inner shape, and a dtype.
      Args:
        row_partitions: A sequence of `RowPartitionSpec`s describing how the
          ragged shape is partitioned.
        static_inner_shape: The static shape of the flat_values.
        dtype: The DType used to encode the shape (tf.int64 or tf.int32).
      """
      # Independent validation and coercion of each argument.
      if not isinstance(row_partitions, Iterable):
        raise TypeError("row_partitions should be an Iterable")
      row_partitions = tuple(row_partitions)
      static_inner_shape = tensor_shape.as_shape(static_inner_shape)
      dtype = dtypes.as_dtype(dtype)
      if not all(isinstance(rp, RowPartitionSpec) for rp in row_partitions):
        raise TypeError(
            "row_partitions should be an Iterable of RowPartitionSpecs")
      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")
      # All fields are now typechecked and internally consistent.
      for spec in row_partitions:
        if spec.dtype != dtype:
          raise ValueError(
              f"dtype of {spec!r} is {spec.dtype!r}: expected {dtype!r}")
      row_partitions = tuple(row_partitions)
      inner_rank = static_inner_shape.rank
      if inner_rank == 0:
        if row_partitions:
          raise ValueError(
              "If row_partitions are provided, must have inner_rank > 0")
      else:
        num_slices_in_dimension = []  # type: Sequence[tensor_shape.Dimension]
        # We first attempt to calculate num_slices_in_dimension through a
        # forward pass, using nrows[k] = nrows[k-1] * uniform_row_length
        # and other tricks.
        for i in range(len(row_partitions)):
          rp = row_partitions[i]
          result = tensor_shape.Dimension(rp.nrows)
          if i > 0:
            previous_rp = row_partitions[i - 1]
            result = result.merge_with(previous_rp.nvals)
            result = result.merge_with(num_slices_in_dimension[-1] *
                                       previous_rp.uniform_row_length)
          num_slices_in_dimension.append(result)
        # In the last step of the forward pass,
        # we combine nvals and the first dimension in static_inner_shape.
        if row_partitions:
          last_rp = row_partitions[-1]
          result = (num_slices_in_dimension[-1] *
                    last_rp.uniform_row_length).merge_with(last_rp.nvals)
          if inner_rank is not None:
            result = result.merge_with(
                tensor_shape.dimension_at_index(static_inner_shape, 0))
            static_inner_shape = result + static_inner_shape[1:]
          num_slices_in_dimension.append(result)
        # Now, we start a backward pass.
        for i in range(len(num_slices_in_dimension) - 1, 0, -1):
          num_slices_in_dimension[i - 1] = num_slices_in_dimension[
              i - 1].merge_with(
                  _safe_floor_div(num_slices_in_dimension[i],
                                  row_partitions[i - 1].uniform_row_length))
        # Finally, we construct the partitions.
        row_partitions = [
            RowPartitionSpec(  # pylint: disable=g-complex-comprehension
                nrows=num_slices_in_dimension[i].value,
                uniform_row_length=rp.uniform_row_length,
                nvals=num_slices_in_dimension[i + 1].value,
                dtype=rp.dtype) for i, rp in enumerate(row_partitions)
        ]
      self._static_inner_shape = static_inner_shape
      self._inner_shape = tensor_spec.TensorSpec([inner_rank], dtype=dtype)
      self._row_partitions = row_partitions
    def __repr__(self):
      return (
          f"DynamicRaggedShape.Spec(row_partitions={self._row_partitions!r}, " +
          f"static_inner_shape={self._static_inner_shape!r}, " +
          f"dtype={self.dtype!r})")
    @classmethod
    def from_value(cls, value: Any) -> "DynamicRaggedShape.Spec":
      """Create a Spec from a DynamicRaggedShape."""
      # super().from_value(...) creates an object, but there is no validation.
      # No methods can be trusted on the object, just the properties.
      initial = super(DynamicRaggedShape.Spec, cls).from_value(value)
      # However, since value is a DynamicRaggedShape, we
      # can guarantee that initial._inner_shape.shape.rank == 1
      # Moreover, if inner_shape.shape[0] is not None, then
      # static_inner_shape.rank is not None.
      return DynamicRaggedShape.Spec(
          row_partitions=initial._row_partitions,
          static_inner_shape=initial._static_inner_shape,
          dtype=initial._inner_shape.dtype)
    # TODO(martinz): it is unclear what the default uniformity of RowPartitions
    # should be, so I am moving this to experimental until we figure it out.
    # Also, while I have specified this is meant to represent a shape of a
    # proper Tensor instead of a RaggedTensor, this is also subject to
    # interpretation.
    @classmethod
    def _from_tensor_shape(cls, shape: Any, num_row_partitions: int,
                           dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      """Creates a `DynamicRaggedShape.Spec` corresponding to a `tf.TensorShape`.
      It is assumed that this is a `tf.TensorShape` coming from a
      `tf.TensorSpec`, not from `RaggedTensor.shape`.
      In addition to the shape, we need to know the number of row partitions,
      and the dtype used in the shape (tf.int32 or tf.int64).
      Within the dimensions that are partitioned, all dimensions are assumed
      to be uniform.
      Args:
        shape: a TensorShape.
        num_row_partitions: the ragged rank of the RaggedShape.
        dtype: the dtype of the shape (not the tensor); tf.int64 or tf.int32.
      Returns:
        a DynamicRaggedShape.Spec representing a TensorShape.
      """
      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")
      shape = tensor_shape.as_shape(shape)
      if shape.rank is None:
        row_partitions = [
            RowPartitionSpec(dtype=dtype) for _ in range(num_row_partitions)
        ]
        return DynamicRaggedShape.Spec(
            row_partitions=row_partitions,
            static_inner_shape=tensor_shape.TensorShape(None),
            dtype=dtype)
      if shape.rank <= 1:
        # Create a scalar or vector shape.
        if num_row_partitions:
          raise ValueError("num_row_partitions should be zero " +
                           "if shape is a scalar or vector.")
        return DynamicRaggedShape.Spec(
            row_partitions=[], static_inner_shape=shape, dtype=dtype)
      if shape.rank <= num_row_partitions:
        raise ValueError("num_row_partitions must be less than rank")
      num_elements_so_far = tensor_shape.dimension_value(shape[0])
      rp_specs = []
      for i in range(num_row_partitions):
        current_dim = tensor_shape.dimension_value(shape[i + 1])
        if current_dim is None or num_elements_so_far is None:
          nvals = None
        else:
          nvals = num_elements_so_far * current_dim
        rp_specs.append(
            RowPartitionSpec(
                nrows=num_elements_so_far,
                nvals=nvals,
                uniform_row_length=current_dim,
                dtype=dtype))
        num_elements_so_far = nvals
      static_inner_shape = tensor_shape.TensorShape(
          [num_elements_so_far]) + shape[num_row_partitions + 1:]
      return DynamicRaggedShape.Spec(
          row_partitions=rp_specs,
          static_inner_shape=static_inner_shape,
          dtype=dtype)
    @classmethod
    def _from_spec(
        cls,
        spec: Union["DynamicRaggedShape.Spec", ragged_tensor.RaggedTensorSpec,
                    tensor_spec.TensorSpec],
        dtype: dtypes.DType = dtypes.int64) -> "DynamicRaggedShape.Spec":
      """Create a TypeSpec for the shape of an object with a given TypeSpec.
      I.e., if `x_spec = tf.type_spec_from_value(x)`, then
      `DynamicRaggedShape.from_spec(x_spec)` returns a TypeSpec compatible with
      `tf.type_spec_from_value(tf.shape(x))`.
      >>> rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
      >>> rt_spec = tf.type_spec_from_value(rt)
      >>> rt_shape = DynamicRaggedShape.from_tensor(rt)
      >>> shape_spec_1 = tf.type_spec_from_value(rt_shape)
      >>> shape_spec_2 = DynamicRaggedShape.Spec._from_spec(rt_spec)
      >>> assert shape_spec_1.is_compatible_with(shape_spec_2)
      Args:
        spec: a Spec of a Tensor or RaggedTensor.
        dtype: the default dtype (if necessary).
      Returns:
        A Spec of the shape of a Tensor or RaggedTensor.
      """
      # TODO(martinz): Add StructuredTensor.Spec when its easy.
      if isinstance(spec, DynamicRaggedShape.Spec):
        return spec
      elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
        return cls._from_tensor_shape(spec.shape, spec.ragged_rank,
                                      spec.row_splits_dtype)
      elif isinstance(spec, tensor_spec.TensorSpec):
        return cls._from_tensor_shape(
            shape=spec.shape, num_row_partitions=0, dtype=dtype)
    @property
    def dtype(self) -> dtypes.DType:
      return self._inner_shape.dtype
    @property
    def inner_rank(self) -> Optional[int]:
      if self._static_inner_shape.rank is not None:
        return self._static_inner_shape.rank
      if self._inner_shape.shape.rank is None:
        return None
      return tensor_shape.dimension_value(self._inner_shape.shape[0])
    @property
    def num_row_partitions(self) -> int:
      return len(self._row_partitions)
    @property
    def rank(self) -> Optional[int]:
      inner_rank = self.inner_rank
      return None if inner_rank is None else inner_rank + self.num_row_partitions
    def _dimension(self, index: int) -> Optional[int]:
      """Get the size of dimension index, if known statically."""
      if index == 0:
        if self._row_partitions:
          return self._row_partitions[0].nrows
        elif self.inner_rank is None:
          return None
        elif self.inner_rank == 0:
          raise ValueError("Index out of range: 0.")
        else:
          return tensor_shape.dimension_value(self._static_inner_shape[0])
      if index <= len(self._row_partitions):
        return self._row_partitions[index - 1].uniform_row_length
      relative_index = index - self.num_row_partitions
      if self.inner_rank is None:
        return None
      elif self.inner_rank <= relative_index:
        raise ValueError(f"Index out of range: {index}.")
      else:
        return tensor_shape.dimension_value(
            self._static_inner_shape[relative_index])
    def _num_slices_in_dimension(self, axis: int) -> Optional[int]:
      """The total size of a dimension (like nvals).
      This is a static version of DynamicRaggedShape._num_slices_in_dimension()
      Example:
      ```
      shape = DynamicRaggedShape.Spec(
        _row_partitions=[
          RowPartitionSpec(nrows=3, nvals=14, dtype=tf.int32)
          RowPartitionSpec(nrows=14, nvals=25, dtype=tf.int32)
        ],
        _static_inner_shape=tf.TensorShape([25, 3, 4]),
        _inner_shape=tf.TensorSpec(tf.TensorShape([3]), dtype=tf.int32))
      shape._num_slices_in_dimension(0) = 3
      shape._num_slices_in_dimension(1) = 14
      shape._num_slices_in_dimension(2) = 25
      shape._num_slices_in_dimension(3) = 3
      shape._num_slices_in_dimension(4) = 4
      shape._num_slices_in_dimension(-2) = 3
      ```
      Args:
        axis: the last dimension to include.
      Returns:
        the number of values in a dimension.
      """
      if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
      axis = array_ops.get_positive_axis(axis, self.rank, ndims_name="rank")
      if axis == 0:
        return self._dimension(0)
      if axis <= self.num_row_partitions:
        # TODO(martinz): use nvals OR nrows, whichever is defined.
        return self._row_partitions[axis - 1].nvals
      remainder = axis - (self.num_row_partitions - 1)
      head_inner_shape = self._static_inner_shape[:remainder]
      return head_inner_shape.num_elements()
    def with_dtype(self, dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      """Return the same spec, but with a different DType."""
      new_rp_specs = [rp.with_dtype(dtype) for rp in self._row_partitions]
      return DynamicRaggedShape.Spec(
          row_partitions=new_rp_specs,
          static_inner_shape=self._static_inner_shape,
          dtype=dtype)
    def _merge_with(
        self, other: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape.Spec":
      """Merges all information between two specs.
      Specs are expected to represent the same information modulo
      num_row_partitons.
      If the specs are of different ranks, then fail.
      Args:
        other: another Spec of the same rank.
      Returns:
        a Spec with the union of information.
      """
      max_num_row_partitions = max(self.num_row_partitions,
                                   other.num_row_partitions)
      a = self._with_num_row_partitions(max_num_row_partitions)
      b = other._with_num_row_partitions(max_num_row_partitions)
      new_rp = [
          a._merge_with(b)
          for (a, b) in zip(a._row_partitions, b._row_partitions)
      ]
      new_static_inner_shape = a._static_inner_shape.merge_with(
          b._static_inner_shape)
      dtype = b.dtype if (a.dtype == dtypes.int32) else dtypes.int64
      return DynamicRaggedShape.Spec(
          new_rp, new_static_inner_shape, dtype=dtype)
    def _with_num_row_partitions(
        self, new_num_row_partitions: int) -> "DynamicRaggedShape.Spec":
      """Change the number of row partitions in the spec."""
      rank = self.rank
      if rank is None:
        raise ValueError(
            "Changing num_row_partitions with unknown rank unsupported")
      if new_num_row_partitions > max(rank - 1, 0):
        raise ValueError("Number of row partitions too large")
      if new_num_row_partitions < 0:
        raise ValueError("Number of row partitions negative")
      if self.num_row_partitions == new_num_row_partitions:
        return self
      elif self.num_row_partitions < new_num_row_partitions:
        # TODO(martinz): Consider swapping.
        rp_delta = new_num_row_partitions - self.num_row_partitions
        tail_shape = DynamicRaggedShape.Spec._from_tensor_shape(
            self._static_inner_shape, rp_delta, self.dtype)
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions + tail_shape._row_partitions,
            static_inner_shape=tail_shape._static_inner_shape,
            dtype=self.dtype)
      else:
        assert self.num_row_partitions > new_num_row_partitions
        new_row_partitions = self._row_partitions[:new_num_row_partitions]
        last_row_partition = new_row_partitions[-1]
        old_row_partitions = self._row_partitions[new_num_row_partitions:]
        new_static_inner_shape = (
            tensor_shape.TensorShape(
                [last_row_partition.nvals] +
                [x.uniform_row_length for x in old_row_partitions]) +
            self._static_inner_shape[1:])
        return DynamicRaggedShape.Spec(new_row_partitions,
                                       new_static_inner_shape, self.dtype)
    def _set_rank_if_unknown(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      """Ensures this has a known rank at least new_rank."""
      if new_rank is None:
        raise TypeError("new_rank is None, but expected int")
      if new_rank < 0:
        raise ValueError("Rank must be non-negative")
      current_rank = self.rank
      if current_rank is not None and current_rank < new_rank:
        raise ValueError(
            "Rank is {current_rank}, expected at least {new_rank}.".format(
                current_rank=current_rank, new_rank=new_rank))
      if current_rank is not None:
        return self
      if self._row_partitions:
        new_inner_rank = max(new_rank - self.num_row_partitions, 1)
        first_dim = self._row_partitions[-1].nvals
        static_inner_shape = tensor_shape.TensorShape([first_dim] + [None] *
                                                      (new_inner_rank - 1))
      else:
        static_inner_shape = tensor_shape.TensorShape([None] * new_rank)
      return DynamicRaggedShape.Spec(
          row_partitions=self._row_partitions,
          static_inner_shape=static_inner_shape,
          dtype=self.dtype)
    def _truncate(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      """Truncate a ragged shape spec.
      For example, if the original spec s was for a shape:
      [3, [4, 1], 2, 7]
      Then truncate_dynamic_ragged_shape_spec(s, 3) is a spec for:
      [3, [4, 1], 2]
      Args:
        new_rank: the new rank
      Returns:
        A truncated DynamicRaggedShape.Spec.
      """
      if self.rank is None:
        return self._set_rank_if_unknown(new_rank)._truncate(new_rank)
      if new_rank == 0:
        return DynamicRaggedShape.Spec._from_tensor_shape([], 0, self.dtype)
      if new_rank == 1:
        vector_size = self._dimension(0)
        return DynamicRaggedShape.Spec._from_tensor_shape([vector_size], 0,
                                                          self.dtype)
      if new_rank < self.num_row_partitions + 1:
        new_row_partitions = self._row_partitions[:new_rank - 1]
        new_static_inner_shape = tensor_shape.TensorShape(
            [new_row_partitions[-1].nvals])
        return DynamicRaggedShape.Spec(
            row_partitions=new_row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)
      else:
        remainder = new_rank - self.num_row_partitions
        new_static_inner_shape = self._static_inner_shape[:remainder]
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)
    def _to_tensor_shape(self):
      """Get a tensor shape corresponding to this type."""
      alt = self
      if alt._static_inner_shape.rank is None:
        return tensor_shape.TensorShape(None)
      if alt._static_inner_shape.rank == 0:
        assert not alt._row_partitions
        return alt._static_inner_shape
      prefix = [alt._dimension(0)]
      prefix.extend([rp.uniform_row_length for rp in alt._row_partitions])
      suffix = alt._static_inner_shape[1:]
      return tensor_shape.TensorShape(prefix) + suffix
