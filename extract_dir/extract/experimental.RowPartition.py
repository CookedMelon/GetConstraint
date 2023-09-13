@tf_export("experimental.RowPartition")
class RowPartition(composite_tensor.CompositeTensor):
  """Partitioning of a sequence of values into contiguous subsequences ("rows").
  A `RowPartition` describes how a sequence with `nvals` items should be
  divided into `nrows` contiguous subsequences ("rows").  For example, a
  `RowPartition` could be used to partition the vector `[1, 2, 3, 4, 5]` into
  subsequences `[[1, 2], [3], [], [4, 5]]`.  Note that `RowPartition` stores
  information about how values are partitioned, but does not include the
  partitioned values themselves.  `tf.RaggedTensor` is used to pair a `values`
  tensor with one or more `RowPartition`s, providing a complete encoding for a
  ragged tensor (i.e. a tensor with variable-length dimensions).
  `RowPartition`s may be defined using several different schemes:
    * `row_lengths`: an integer vector with shape `[nrows]`, which specifies
      the length of each row.
    * `row_splits`: an integer vector with shape `[nrows+1]`, specifying the
      "split points" between each row.
    * `row_starts`: an integer vector with shape `[nrows]`, which specifies
      the start offset for each row.  Equivalent to `row_splits[:-1]`.
    * `row_limits`: an integer vector with shape `[nrows]`, which specifies
      the stop offset for each row.  Equivalent to `row_splits[1:]`.
    * `value_rowids` is an integer vector with shape `[nvals]`, corresponding
      one-to-one with sequence values, which specifies the row that each value
      belongs to.  If the partition has empty trailing rows, then `nrows`
      must also be specified.
    * `uniform_row_length` is an integer scalar, specifying the length of every
      row.  This scheme may only be used if all rows have the same length.
  For example, the following `RowPartition`s all represent the partitioning of
  8 values into 5 sublists as follows: `[[*, *, *, *], [], [*, *, *], [*], []]`.
  >>> p1 = RowPartition.from_row_lengths([4, 0, 3, 1, 0])
  >>> p2 = RowPartition.from_row_splits([0, 4, 4, 7, 8, 8])
  >>> p3 = RowPartition.from_row_starts([0, 4, 4, 7, 8], nvals=8)
  >>> p4 = RowPartition.from_row_limits([4, 4, 7, 8, 8])
  >>> p5 = RowPartition.from_value_rowids([0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  For more information about each scheme, see the documentation for the
  its factory method.  For additional examples, see the documentation on
  `tf.RaggedTensor`.
  ### Precomputed Encodings
  `RowPartition` always stores at least one encoding of the partitioning, but
  it can be configured to cache additional encodings as well.  This can
  avoid unnecessary recomputation in eager mode.  (In graph mode, optimizations
  such as common subexpression elimination will typically prevent these
  unnecessary recomputations.)  To check which encodings are precomputed, use
  `RowPartition.has_precomputed_<encoding>`.  To cache an additional
  encoding, use `RowPartition.with_precomputed_<encoding>`.
  """
  # =============================================================================
  # Constructor (private)
  # =============================================================================
  def __init__(self,
               row_splits,
               row_lengths=None,
               value_rowids=None,
               nrows=None,
               uniform_row_length=None,
               nvals=None,
               internal=False):
    """Creates a `RowPartition` from the specified encoding tensor(s).
    This constructor is private -- please use one of the following ops to
    build `RowPartition`s:
      * `RowPartition.from_row_lengths`
      * `RowPartition.from_value_rowids`
      * `RowPartition.from_row_splits`
      * `RowPartition.from_row_starts`
      * `RowPartition.from_row_limits`
      * `RowPartition.from_uniform_row_length`
    If row_splits is has a constant value, then all other arguments should
    have a constant value.
    Args:
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.
      row_lengths: A 1-D integer tensor with shape `[nrows]`
      value_rowids: A 1-D integer tensor with shape `[nvals]`.
      nrows: A 1-D integer scalar tensor.
      uniform_row_length: A scalar tensor.
      nvals: A scalar tensor.
      internal: Private key value, required to ensure that this private
        constructor is *only* called from the factory methods.
    Raises:
      TypeError: If a row partitioning tensor has an inappropriate dtype.
      TypeError: If exactly one row partitioning argument was not specified.
      ValueError: If a row partitioning tensor has an inappropriate shape.
      ValueError: If multiple partitioning arguments are specified.
      ValueError: If nrows is specified but value_rowids is not None.
    """
    if internal is not _row_partition_factory_key:
      raise ValueError("RowPartition constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "RowPartition.from_row_lengths())")
    # Validate the arguments.
    if not isinstance(row_splits, ops.Tensor):
      raise TypeError("Row-partitioning argument must be a Tensor, got %r" %
                      row_splits)
    if row_splits.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("Row-partitioning argument must be int32 or int64")
    # Validate shapes & dtypes.
    row_splits.shape.assert_has_rank(1)
    row_splits.set_shape([None])
    self._row_splits = row_splits
    # Store any cached tensors.  These are used to avoid unnecessary
    # round-trip conversions when a RowPartition is constructed from
    # lengths or rowids, and we later want those lengths/rowids back.
    for tensor in [row_lengths, value_rowids, nrows, uniform_row_length, nvals]:
      if tensor is not None:
        if not isinstance(tensor, ops.Tensor):
          raise TypeError("Cached value must be a Tensor or None.")
        elif tensor.dtype != row_splits.dtype:
          raise ValueError(f"Inconsistent dtype for encoding tensors: "
                           f"{tensor} vs {row_splits}")
    self._row_lengths = row_lengths
    self._value_rowids = value_rowids
    self._nrows = nrows
    self._uniform_row_length = uniform_row_length
    self._nvals = nvals
  # =============================================================================
  # Factory Methods
  # =============================================================================
  @classmethod
  def from_value_rowids(cls,
                        value_rowids,
                        nrows=None,
                        validate=True,
                        dtype=None,
                        dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `value_rowids`.
    This `RowPartition` divides a sequence `values` into rows by specifying
    which row each value should be added to:
    ```python
    partitioned_rows = [[] for _ in nrows]
    for (value, rowid) in zip(values, value_rowids):
      partitioned_rows[rowid].append(value)
    ```
    Args:
      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An integer scalar specifying the number of rows.  This should be
        specified if the `RowPartition` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or greater than or equal to zero if
        `value_rowids` is empty). Defaults to `value_rowids[-1] + 1` (or zero if
        `value_rowids` is empty).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `value_rowids`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.
    #### Example:
    >>> print(RowPartition.from_value_rowids(
    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
    ...     nrows=4))
    tf.RowPartition(row_splits=[0 4 4 7 8])
    """
    # Local import bincount_ops to avoid import-cycle since bincount_ops
    # imports ragged_tensor.
    from tensorflow.python.ops import bincount_ops  # pylint: disable=g-import-not-at-top
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(None, "RowPartitionFromValueRowIds",
                        [value_rowids, nrows]):
      value_rowids = cls._convert_row_partition(
          value_rowids, "value_rowids", dtype_hint=dtype_hint, dtype=dtype)
      if nrows is None:
        const_rowids = tensor_util.constant_value(value_rowids)
        if const_rowids is None:
          nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
          const_nrows = None
        else:
          const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
          nrows = ops.convert_to_tensor(
              const_nrows, value_rowids.dtype, name="nrows")
      else:
        nrows = ops.convert_to_tensor(nrows, value_rowids.dtype, "nrows")
        const_nrows = tensor_util.constant_value(nrows)
        if const_nrows is not None:
          if const_nrows < 0:
            raise ValueError("Expected nrows >= 0; got %d" % const_nrows)
          const_rowids = tensor_util.constant_value(value_rowids)
          if const_rowids is not None and const_rowids.size > 0:
            if not const_nrows >= const_rowids[-1] + 1:
              raise ValueError(
                  "Expected nrows >= value_rowids[-1] + 1; got nrows=%d, "
                  "value_rowids[-1]=%d" % (const_nrows, const_rowids[-1]))
      value_rowids.shape.assert_has_rank(1)
      nrows.shape.assert_has_rank(0)
      if validate:
        msg = ("Arguments to from_value_rowids do not form a valid "
               "RowPartition")
        checks = [
            check_ops.assert_rank(value_rowids, 1, message=msg),
            check_ops.assert_rank(nrows, 0, message=msg),
            check_ops.assert_non_negative(value_rowids[:1], message=msg),
            _assert_monotonic_increasing(value_rowids, message=msg),
            check_ops.assert_less(value_rowids[-1:], nrows, message=msg),
        ]
        value_rowids = control_flow_ops.with_dependencies(checks, value_rowids)
      # Convert value_rowids & nrows to row_splits.
      # Note: we don't use segment_ids_to_row_splits() here because we want
      # to save the intermediate value `row_lengths`, so we can cache it.
      # TODO(b/116708836) Upgrade bincount to accept int64 so we can skip the
      # cast.
      value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
      nrows_int32 = math_ops.cast(nrows, dtypes.int32)
      row_lengths = bincount_ops.bincount(
          value_rowids_int32,
          minlength=nrows_int32,
          maxlength=nrows_int32,
          dtype=value_rowids.dtype)
      row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
      if const_nrows is not None:
        row_lengths.set_shape([const_nrows])
        row_splits.set_shape([const_nrows + 1])
      return cls(
          row_splits=row_splits,
          row_lengths=row_lengths,
          value_rowids=value_rowids,
          nrows=nrows,
          internal=_row_partition_factory_key)
  @classmethod
  def from_row_splits(cls,
                      row_splits,
                      validate=True,
                      dtype=None,
                      dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `row_splits`.
    This `RowPartition` divides a sequence `values` into rows by indicating
    where each row begins and ends:
    ```python
    partitioned_rows = []
    for i in range(len(row_splits) - 1):
      row_start = row_splits[i]
      row_end = row_splits[i + 1]
      partitioned_rows.append(values[row_start:row_end])
    ```
    Args:
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `row_splits`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    Raises:
      ValueError: If `row_splits` is an empty list.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    if isinstance(row_splits, (list, tuple)) and not row_splits:
      raise ValueError("row_splits tensor may not be empty.")
    if isinstance(row_splits, tensor_spec.TensorSpec):
      return cls(row_splits=row_splits, internal=_row_partition_factory_key)
    with ops.name_scope(None, "RowPartitionFromRowSplits", [row_splits]):
      row_splits = cls._convert_row_partition(
          row_splits, "row_splits", dtype_hint=dtype_hint, dtype=dtype)
      row_splits.shape.assert_has_rank(1)
      if validate:
        msg = "Arguments to from_row_splits do not form a valid RaggedTensor:"
        checks = [
            check_ops.assert_rank(row_splits, 1, message=(msg + "rank")),
            _assert_zero(row_splits[0], message=(msg + "zero")),
            _assert_monotonic_increasing(
                row_splits, message=(msg + "monotonic")),
        ]
        row_splits = control_flow_ops.with_dependencies(checks, row_splits)
      return cls(row_splits=row_splits, internal=_row_partition_factory_key)
  @classmethod
  def from_row_lengths(cls,
                       row_lengths,
                       validate=True,
                       dtype=None,
                       dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `row_lengths`.
    This `RowPartition` divides a sequence `values` into rows by indicating
    the length of each row:
    ```python
    partitioned_rows = [[values.pop(0) for _ in range(length)]
                        for length in row_lengths]
    ```
    Args:
      row_lengths: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `row_lengths`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(None, "RowPartitionFromRowLengths", [row_lengths]):
      row_lengths = cls._convert_row_partition(
          row_lengths, "row_lengths", dtype_hint=dtype_hint, dtype=dtype)
      row_lengths.shape.assert_has_rank(1)
      if validate:
        msg = "Arguments to from_row_lengths do not form a valid RowPartition"
        checks = [
            check_ops.assert_rank(row_lengths, 1, message=msg),
            check_ops.assert_non_negative(row_lengths, message=msg),
        ]
        row_lengths = control_flow_ops.with_dependencies(checks, row_lengths)
      row_limits = math_ops.cumsum(row_lengths)
      row_splits = array_ops.concat([[0], row_limits], axis=0)
      return cls(
          row_splits=row_splits,
          row_lengths=row_lengths,
          internal=_row_partition_factory_key)
  @classmethod
  def from_row_starts(cls,
                      row_starts,
                      nvals,
                      validate=True,
                      dtype=None,
                      dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `row_starts`.
    Equivalent to: `from_row_splits(concat([row_starts, nvals], axis=0))`.
    Args:
      row_starts: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative and sorted in ascending order.  If `nrows>0`, then
        `row_starts[0]` must be zero.
      nvals: A scalar tensor indicating the number of values.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `row_starts`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(None, "RowPartitionFromRowStarts", [row_starts]):
      row_starts = cls._convert_row_partition(
          row_starts, "row_starts", dtype_hint=dtype_hint, dtype=dtype)
      row_starts.shape.assert_has_rank(1)
      # TODO(martinz): nvals and row_starts could be inconsistent at call time,
      # even though they eventually end up the same type.
      nvals = math_ops.cast(nvals, row_starts.dtype)
      if validate:
        msg = "Arguments to from_row_starts do not form a valid RaggedTensor"
        checks = [
            check_ops.assert_rank(row_starts, 1, message=msg),
            _assert_zero(row_starts[:1], message=msg),
            _assert_monotonic_increasing(row_starts, message=msg),
            check_ops.assert_less_equal(row_starts[-1:], nvals, message=msg),
        ]
        row_starts = control_flow_ops.with_dependencies(checks, row_starts)
      row_splits = array_ops.concat([row_starts, [nvals]], axis=0)
      return cls(row_splits=row_splits, nvals=nvals,
                 internal=_row_partition_factory_key)
  @classmethod
  def from_row_limits(cls,
                      row_limits,
                      validate=True,
                      dtype=None,
                      dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `row_limits`.
    Equivalent to: `from_row_splits(values, concat([0, row_limits], axis=0))`.
    Args:
      row_limits: A 1-D integer tensor with shape `[nrows]`.  Must be sorted in
        ascending order.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `row_limits`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(None, "RowPartitionFromRowLimits", [row_limits]):
      row_limits = cls._convert_row_partition(
          row_limits, "row_limits", dtype_hint=dtype_hint, dtype=dtype)
      row_limits.shape.assert_has_rank(1)
      if validate:
        msg = "Arguments to from_row_limits do not form a valid RaggedTensor"
        checks = [
            check_ops.assert_rank(row_limits, 1, message=msg),
            check_ops.assert_non_negative(row_limits[:1], message=msg),
            _assert_monotonic_increasing(row_limits, message=msg),
        ]
        row_limits = control_flow_ops.with_dependencies(checks, row_limits)
      zero = array_ops.zeros([1], row_limits.dtype)
      row_splits = array_ops.concat([zero, row_limits], axis=0)
      return cls(row_splits=row_splits, internal=_row_partition_factory_key)
  @classmethod
  def from_uniform_row_length(cls,
                              uniform_row_length,
                              nvals=None,
                              nrows=None,
                              validate=True,
                              dtype=None,
                              dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `uniform_row_length`.
    This `RowPartition` divides a sequence `values` into rows that all have
    the same length:
    ```python
    partitioned_rows = [[values.pop(0) for _ in range(uniform_row_length)]
             for _ in range(nrows)]
    ```
    Note that either or both of nvals and nrows must be specified.
    Args:
      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The
        size of the outer axis of `values` must be evenly divisible by
        `uniform_row_length`.
      nvals: a non-negative scalar integer tensor for the number of values.
        Must be specified if nrows is not specified. If not specified,
        defaults to uniform_row_length*nrows
      nrows: The number of rows in the constructed RowPartition.  If not
        specified, then it defaults to `nvals/uniform_row_length` (or `0` if
        `uniform_row_length==0`).  `nrows` only needs to be specified if
        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be
        `nvals`.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `uniform_row_length`, dtype_hint,
        or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    if nrows is None and nvals is None:
      raise ValueError("Either (or both) of nvals and nrows must be specified")
    with ops.name_scope(None, "RowPartitionFromUniformRowLength",
                        [uniform_row_length, nrows]):
      [uniform_row_length, nvals, nrows
      ] = _convert_all_to_tensors([(uniform_row_length, "uniform_row_length"),
                                   (nvals, "nvals"), (nrows, "nrows")],
                                  dtype=dtype,
                                  dtype_hint=dtype_hint)
      uniform_row_length.shape.assert_has_rank(0)
      # Find nrows.
      const_row_length = tensor_util.constant_value(uniform_row_length)
      if nrows is None:
        if const_row_length is None:
          # Avoid division by zero if uniform_row_length==0 (and nvals==0).
          rowlen_or_1 = math_ops.maximum(
              uniform_row_length,
              constant_op.constant(1, uniform_row_length.dtype))
          nrows = nvals // rowlen_or_1
        elif const_row_length == 0:
          nrows = constant_op.constant(0, dtype=uniform_row_length.dtype)
        else:
          nrows = nvals // const_row_length
      const_nrows = None if nrows is None else tensor_util.constant_value(nrows)
      const_nvals = None if nvals is None else tensor_util.constant_value(nvals)
      const_uniform_row_length = tensor_util.constant_value(uniform_row_length)
      checks = []
      if const_nvals is None and const_nrows is not None and const_uniform_row_length is not None:
        const_nvals = const_nrows * const_uniform_row_length
        if nvals is not None and validate:
          checks.append(check_ops.assert_equal(nvals, const_nvals))
        nvals = constant_op.constant(const_nvals, uniform_row_length.dtype)
      if nvals is None:
        nvals = nrows * uniform_row_length
      # Find row_splits.
      if const_nrows is not None and const_row_length is not None:
        row_splits = [v * const_row_length for v in range(const_nrows + 1)]
        row_splits = constant_op.constant(row_splits, uniform_row_length.dtype)
      else:
        row_splits = math_ops.range(
            nrows + 1, dtype=uniform_row_length.dtype) * uniform_row_length
      if validate:
        if (const_nrows is None or const_row_length is None or
            const_nvals is None):
          checks.append(
              check_ops.assert_equal(
                  nrows * uniform_row_length, nvals,
                  ("uniform_row_length", uniform_row_length, "times nrows",
                   nrows, "must equal nvals", nvals)))
        else:
          if const_nrows * const_row_length != const_nvals:
            raise ValueError(
                "uniform_row_length=%d times nrows=%d must equal nvals=%d" %
                (const_row_length, const_nrows, const_nvals))
        if uniform_row_length.shape.rank is None:
          checks.append(
              check_ops.assert_rank(
                  uniform_row_length,
                  0,
                  message="uniform_row_length must be a scalar."))
        const_row_length = tensor_util.constant_value(uniform_row_length)
        if const_row_length is None:
          checks.append(
              check_ops.assert_greater_equal(
                  uniform_row_length,
                  constant_op.constant(0, uniform_row_length.dtype),
                  message="uniform_row_length must be >= 0."))
        else:
          if const_row_length < 0:
            raise ValueError("uniform_row_length must be >= 0.")
        row_splits = control_flow_ops.with_dependencies(checks, row_splits)
      return cls(
          row_splits=row_splits,
          uniform_row_length=uniform_row_length,
          nrows=nrows,
          nvals=nvals,
          internal=_row_partition_factory_key)
  @classmethod
  def _convert_row_partition(cls, partition, name, dtype=None, dtype_hint=None):
    """Converts `partition` to Tensors.
    Args:
      partition: A row-partitioning tensor for the `RowPartition` being
        constructed.  I.e., one of: row_splits, row_lengths, row_starts,
        row_limits, value_rowids, uniform_row_length.
      name: The name of the row-partitioning tensor.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `uniform_row_length`, dtype_hint,
        or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.
    Returns:
      A tensor equivalent to partition.
    Raises:
      ValueError: if dtype is not int32 or int64.
    """
    if dtype_hint is None:
      dtype_hint = dtypes.int64
    if (isinstance(partition, np.ndarray) and
        partition.dtype == np.int32 and dtype is None):
      partition = ops.convert_to_tensor(partition, name=name)
    else:
      partition = tensor_conversion.convert_to_tensor_v2(
          partition, dtype_hint=dtype_hint, dtype=dtype, name=name
      )
    if partition.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("%s must have dtype int32 or int64" % name)
    return partition
  def _with_dependencies(self, dependencies):
    """Returns a new RowPartition equal to self with control dependencies.
    Specifically, self._row_splits is gated by the given control dependencies.
    Used to add sanity checks to the constructors.
    Args:
      dependencies: a list of tensors to use as dependencies.
    Returns:
      A new RowPartition object.
    """
    new_row_splits = control_flow_ops.with_dependencies(dependencies,
                                                        self._row_splits)
    return RowPartition(
        row_splits=new_row_splits,
        row_lengths=self._row_lengths,
        value_rowids=self._value_rowids,
        nrows=self._nrows,
        uniform_row_length=self._uniform_row_length,
        internal=_row_partition_factory_key)
  # =============================================================================
  # Accessors
  # =============================================================================
  @property
  def dtype(self):
    """The `DType` used to encode the row partition (either int32 or int64)."""
    return self._row_splits.dtype
  def row_splits(self):
    """Returns the row-split indices for this row partition.
    `row_splits` specifies where the values for each row begin and end.
    In particular, the values for row `i` are stored in the slice
    `values[row_splits[i]:row_splits[i+1]]`.
    Returns:
      A 1-D integer `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits()[0] == 0`.
      `self.row_splits()[-1] == self.nvals()`.
    """
    return self._row_splits
  def value_rowids(self):
    """Returns the row indices for this row partition.
    `value_rowids` specifies the row index fo reach value.  In particular,
    `value_rowids[i]` is the row index for `values[i]`.
    Returns:
      A 1-D integer `Tensor` with shape `[self.nvals()]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
    """
    if self._value_rowids is not None:
      return self._value_rowids
    return segment_id_ops.row_splits_to_segment_ids(self._row_splits)
  def nvals(self):
    """Returns the number of values partitioned by this `RowPartition`.
    If the sequence partitioned by this `RowPartition` is a tensor, then
    `nvals` is the size of that tensor's outermost dimension -- i.e.,
    `nvals == values.shape[0]`.
    Returns:
      scalar integer Tensor
    """
    # TODO(martinz): Uncomment these lines.
    # if self._nvals is not None:
    #   return self._nvals
    return self._row_splits[-1]
  def nrows(self):
    """Returns the number of rows created by this `RowPartition`.
    Returns:
      scalar integer Tensor
    """
    if self._nrows is not None:
      return self._nrows
    nsplits = tensor_shape.dimension_at_index(self._row_splits.shape, 0)
    if nsplits.value is None:
      return array_ops.shape(self._row_splits, out_type=self.dtype)[0] - 1
    else:
      return constant_op.constant(nsplits.value - 1, dtype=self.dtype)
  def uniform_row_length(self):
    """Returns the length of each row in this partition, if rows are uniform.
    If all rows in this `RowPartition` have the same length, then this returns
    that length as a scalar integer `Tensor`.  Otherwise, it returns `None`.
    Returns:
      scalar Tensor with `type=self.dtype`, or `None`.
    """
    return self._uniform_row_length
  def row_starts(self):
    """Returns the start indices for rows in this row partition.
    These indices specify where the values for each row begin.
    `partition.row_starts()` is equal to `partition.row_splits()[:-1]`.
    Returns:
      A 1-D integer Tensor with shape `[self.nrows()]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
      `self.row_starts()[0] == 0`.
      `self.row_starts()[-1] <= self.nvals()`.
    """
    return self._row_splits[:-1]
  def row_limits(self):
    """Returns the limit indices for rows in this row partition.
    These indices specify where the values for each row end.
    `partition.row_limits()` is equal to `partition.row_splits()[:-1]`.
    Returns:
      A 1-D integer Tensor with shape `[self.nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
      `self.row_limits()[-1] == self.nvals()`.
    """
    return self._row_splits[1:]
  def row_lengths(self):
    """Returns the lengths of rows in this `RowPartition`.
    Returns:
      A 1-D integer Tensor with shape `[self.nrows]`.
      The returned tensor is nonnegative.
      `tf.reduce_sum(self.row_lengths) == self.nvals()`.
    """
    if self._row_lengths is not None:
      return self._row_lengths
    splits = self._row_splits
    return splits[1:] - splits[:-1]
  @property
  def static_nrows(self):
    """The number of rows in this partition, if statically known.
    ```python
    self.row_lengths().shape == [self.static_nrows]
    self.row_starts().shape == [self.static_nrows]
    self.row_limits().shape == [self.static_nrows]
    self.row_splits().shape == [self.static_nrows + 1]
    ```
    Returns:
      The number of rows in this partition as an `int` (if statically known);
      or `None` (otherwise).
    """
    if self._row_splits is not None:
      nrows_plus_one = tensor_shape.dimension_value(self._row_splits.shape[0])
      if nrows_plus_one is not None:
        return nrows_plus_one - 1
    if self._row_lengths is not None:
      nrows = tensor_shape.dimension_value(self._row_lengths.shape[0])
      if nrows is not None:
        return nrows
    if self._nrows is not None:
      return tensor_util.constant_value(self._nrows)
    return None
  @property
  def static_nvals(self):
    """The number of values in this partition, if statically known.
    ```python
    self.value_rowids().shape == [self.static_vals]
    ```
    Returns:
      The number of values in this partition as an `int` (if statically known);
      or `None` (otherwise).
    """
    if self._nvals is not None:
      nvals = tensor_util.constant_value(self._nvals)
      if nvals is not None:
        return nvals
    if self._value_rowids is not None:
      nvals = tensor_shape.dimension_at_index(self._value_rowids.shape, 0)
      if nvals.value is not None:
        return nvals.value
    return None
  @property
  def static_uniform_row_length(self):
    """The number of values in each row of this partition, if statically known.
    Returns:
      The number of values in each row of this partition as an `int` (if
      statically known); or `None` (otherwise).
    """
    if self._uniform_row_length is not None:
      return tensor_util.constant_value(self._uniform_row_length)
    return None
  def offsets_in_rows(self):
    """Return the offset of each value.
    RowPartition takes an array x and converts it into sublists.
    offsets[i] is the index of x[i] in its sublist.
    Given a shape, such as:
    [*,*,*],[*,*],[],[*,*]
    This returns:
    0,1,2,0,1,0,1
    Returns:
      an offset for every value.
    """
    return gen_ragged_math_ops.ragged_range(
        starts=constant_op.constant(0, self.dtype),
        limits=self.row_lengths(),
        deltas=constant_op.constant(1, self.dtype)).rt_dense_values
  def is_uniform(self):
    """Returns true if the partition is known to be uniform statically.
    This is based upon the existence of self._uniform_row_length. For example:
    RowPartition.from_row_lengths([3,3,3]).is_uniform()==false
    RowPartition.from_uniform_row_length(5, nvals=20).is_uniform()==true
    RowPartition.from_row_lengths([2,0,2]).is_uniform()==false
    Returns:
      Whether a RowPartition is known to be uniform statically.
    """
    return self._uniform_row_length is not None
  def _static_check(self):
    """Checks if the object is internally consistent.
    Raises:
      ValueError if inconsistent.
    """
    my_dtype = self.dtype
    if self._uniform_row_length is not None:
      if self._uniform_row_length.dtype != my_dtype:
        raise ValueError("_uniform_row_length.dtype=" +
                         str(self._uniform_row_length.dtype) + ", not " +
                         str(my_dtype))
    if self._row_lengths is not None and self._row_lengths.dtype != my_dtype:
      raise ValueError("_row_lengths.dtype=" + str(self._row_lengths.dtype) +
                       ", not " + str(my_dtype))
    if self._value_rowids is not None and self._value_rowids.dtype != my_dtype:
      raise ValueError("_value_rowids.dtype=" + str(self._value_rowids.dtype) +
                       ", not " + str(my_dtype))
    if self._nrows is not None and self._nrows.dtype != my_dtype:
      raise ValueError("_nrows.dtype=" + str(self._nrows.dtype) + ", not " +
                       str(my_dtype))
  # =============================================================================
  # Transformation
  # =============================================================================
  def with_dtype(self, dtype):
    """Returns a copy of this RowPartition with the given encoding dtype.
    Args:
      dtype: The dtype for encoding tensors, such as `row_splits` and `nrows`.
      One of `tf.int32` or `tf.int64`.
    Returns:
      A copy of this RowPartition, with the encoding tensors cast to the given
      type.
    """
    dtype = dtypes.as_dtype(dtype)
    if dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("dtype must be int32 or int64")
    if self.dtype == dtype:
      return self
    return RowPartition(
        row_splits=_cast_if_not_none(self._row_splits, dtype),
        row_lengths=_cast_if_not_none(self._row_lengths, dtype),
        value_rowids=_cast_if_not_none(self._value_rowids, dtype),
        nrows=_cast_if_not_none(self._nrows, dtype),
        uniform_row_length=_cast_if_not_none(self._uniform_row_length, dtype),
        internal=_row_partition_factory_key)
  # =============================================================================
  # String Encoding
  # =============================================================================
  def __repr__(self):
    if self._uniform_row_length is not None:
      return (f"tf.RowPartition(nrows={self._nrows}, "
              f"uniform_row_length={self._uniform_row_length})")
    else:
      return f"tf.RowPartition(row_splits={self._row_splits})"
  # =============================================================================
  # Precomputed Encodings
  # =============================================================================
  def _has_precomputed_row_splits(self):
    """Returns true if `row_splits` has already been computed.
    If true, then `self.row_splits()` will return its value without calling
    any TensorFlow ops.
    """
    return self._row_splits is not None
  def _has_precomputed_row_lengths(self):
    """Returns true if `row_lengths` has already been computed.
    If true, then `self.row_lengths()` will return its value without calling
    any TensorFlow ops.
    """
    return self._row_lengths is not None
  def _has_precomputed_value_rowids(self):
    """Returns true if `value_rowids` has already been computed.
    If true, then `self.value_rowids()` will return its value without calling
    any TensorFlow ops.
    """
    return self._value_rowids is not None
  def _has_precomputed_nrows(self):
    """Returns true if `nrows` has already been computed.
    If true, then `self.nrows()` will return its value without calling
    any TensorFlow ops.
    """
    return self._nrows is not None
  def _has_precomputed_nvals(self):
    """Returns true if `nvals` has already been computed.
    If true, then `self.nvals()` will return its value without calling
    any TensorFlow ops.
    """
    return self._nvals is not None
  def _with_precomputed_row_splits(self):
    """Returns a copy of `self` with `row_splits` precomputed."""
    return RowPartition(
        row_splits=self.row_splits(),
        row_lengths=self._row_lengths,
        value_rowids=self._value_rowids,
        nrows=self._nrows,
        uniform_row_length=self._uniform_row_length,
        nvals=self._nvals,
        internal=_row_partition_factory_key)
  def _with_precomputed_row_lengths(self):
    """Returns a copy of `self` with `row_lengths` precomputed."""
    return RowPartition(
        row_splits=self._row_splits,
        row_lengths=self.row_lengths(),
        value_rowids=self._value_rowids,
        nrows=self._nrows,
        nvals=self._nvals,
        uniform_row_length=self._uniform_row_length,
        internal=_row_partition_factory_key)
  def _with_precomputed_value_rowids(self):
    """Returns a copy of `self` with `value_rowids` precomputed."""
    return RowPartition(
        row_splits=self._row_splits,
        row_lengths=self._row_lengths,
        value_rowids=self.value_rowids(),
        nrows=self._nrows,
        nvals=self._nvals,
        uniform_row_length=self._uniform_row_length,
        internal=_row_partition_factory_key)
  def _with_precomputed_nrows(self):
    """Returns a copy of `self` with `nrows` precomputed."""
    return RowPartition(
        row_splits=self._row_splits,
        row_lengths=self._row_lengths,
        value_rowids=self._value_rowids,
        nrows=self.nrows(),
        nvals=self._nvals,
        uniform_row_length=self._uniform_row_length,
        internal=_row_partition_factory_key)
  def _with_precomputed_nvals(self):
    """Returns a copy of `self` with `row_splits` precomputed."""
    return RowPartition(
        row_splits=self.row_splits(),
        row_lengths=self._row_lengths,
        value_rowids=self._value_rowids,
        nrows=self._nrows,
        nvals=self.nvals(),
        uniform_row_length=self._uniform_row_length,
        internal=_row_partition_factory_key)
  def _merge_with_spec(self, b):
    """Merge with a TypeSpec to create a new RowPartition."""
    a_spec = self._type_spec
    if not a_spec.is_compatible_with(b):
      # TODO(martinz): Should a dynamic check be used here?
      raise ValueError("RowPartition and RowPartitionSpec are not compatible")
    nrows = constant_op.constant(
        b.nrows, self.dtype) if b.nrows is not None else self._nrows
    nvals = constant_op.constant(
        b.nvals, self.dtype) if b.nvals is not None else self._nvals
    uniform_row_length = constant_op.constant(
        b.uniform_row_length, self.dtype
    ) if b.uniform_row_length is not None else self._uniform_row_length
    return RowPartition(
        row_splits=self._row_splits,
        row_lengths=self._row_lengths,
        value_rowids=self._value_rowids,
        nvals=nvals,
        uniform_row_length=uniform_row_length,
        nrows=nrows,
        internal=_row_partition_factory_key)
  def _merge_precomputed_encodings(self, other, validate=True):
    """Returns a RowPartition that merges encodings from `self` and `other`.
    Requires that `self` and `other` describe the same partition.
    Args:
      other: A `RowPartition` that encodes the same partition as `self`.
      validate: If true, then add runtime checks to verify that `self` and
        `other` encode the same row partition.
    Returns:
      A `RowPartition`.
    """
    # pylint: disable=protected-access
    if (self is other or  # Fast path if row partitions are equal.
        (self._row_splits is other._row_splits and
         self._row_lengths is other._row_lengths and
         self._value_rowids is other._value_rowids and
         self._nrows is other._nrows and
         self._nvals is other._nvals and
         self._uniform_row_length is other._uniform_row_length)):
      return self
    # Merge the component tensors.  We only need to validate one encoding.
    # We merge less-expensive encodings first (to avoid expensive validation).
    nrows, nrows_validated = _merge_tensors(self._nrows, other._nrows, "nrows",
                                            validate)
    nvals, _ = _merge_tensors(self._nvals, other._nvals, "nvals", validate)
    uniform_row_length, uniform_row_length_validated = _merge_tensors(
        self._uniform_row_length, other._uniform_row_length,
        "uniform_row_length", validate)
    if uniform_row_length_validated and nrows_validated:
      validate = False  # Validation complete.
    row_splits, row_splits_validated = _merge_tensors(self._row_splits,
                                                      other._row_splits,
                                                      "row_splits", validate)
    if row_splits_validated:
      validate = False  # Validation complete.
    row_lengths, row_lengths_validated = _merge_tensors(self._row_lengths,
                                                        other._row_lengths,
                                                        "row_lengths", validate)
    if row_lengths_validated:
      validate = False  # Validation complete.
    value_rowids, value_rowids_validated = _merge_tensors(
        self._value_rowids, other._value_rowids, "value_rowids", validate)
    if value_rowids_validated and nrows_validated:
      validate = False  # Validation complete.
    # TODO(edloper): If we make the row_splits encoding optional, then there
    # will be cases where we need to do validation at this point -- e.g. if
    # self has only row_splits and other has only value_rowids.  But for
    # now, we are guaranteed to have done validation by this point.
    # Avoid creating new RowPartition objects if we don't need to.
    if (row_splits is self._row_splits and row_lengths is self._row_lengths and
        value_rowids is self._value_rowids and nrows is self._nrows and
        uniform_row_length is self._uniform_row_length):
      return self
    if (row_splits is other._row_splits and
        row_lengths is other._row_lengths and
        value_rowids is other._value_rowids and nrows is other._nrows and
        uniform_row_length is other._uniform_row_length):
      return other
    return RowPartition(
        row_splits=row_splits,
        row_lengths=row_lengths,
        value_rowids=value_rowids,
        nrows=nrows,
        uniform_row_length=uniform_row_length,
        nvals=nvals,
        internal=_row_partition_factory_key)
  # =============================================================================
  # Composite Tensor
  # =============================================================================
  @property
  def _type_spec(self):
    return RowPartitionSpec.from_value(self)
