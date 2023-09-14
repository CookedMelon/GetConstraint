@tf_export(
    v1=["sparse.SparseConditionalAccumulator", "SparseConditionalAccumulator"])
class SparseConditionalAccumulator(ConditionalAccumulatorBase):
  """A conditional accumulator for aggregating sparse gradients.
  Sparse gradients are represented by `IndexedSlices`.
  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.
  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
  """
  def __init__(self,
               dtype,
               shape=None,
               shared_name=None,
               name="sparse_conditional_accumulator",
               reduction_type="MEAN"):
    accumulator_ref = gen_data_flow_ops.sparse_conditional_accumulator(
        dtype=dtype,
        shape=shape,
        shared_name=shared_name,
        name=name,
        reduction_type=reduction_type)
    super(SparseConditionalAccumulator, self).__init__(dtype, shape,
                                                       accumulator_ref)
  def apply_indexed_slices_grad(self, grad, local_step=0, name=None):
    """Attempts to apply a gradient to the accumulator.
    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.
    Args:
      grad: The gradient `IndexedSlices` to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.
    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.
    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    return self.apply_grad(
        grad_indices=grad.indices,
        grad_values=grad.values,
        grad_shape=grad.dense_shape,
        local_step=local_step,
        name=name)
  def apply_grad(self,
                 grad_indices,
                 grad_values,
                 grad_shape=None,
                 local_step=0,
                 name=None):
    """Attempts to apply a sparse gradient to the accumulator.
    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.
    A sparse gradient is represented by its indices, values and possibly empty
    or None shape. Indices must be a vector representing the locations of
    non-zero entries in the tensor. Values are the non-zero slices of the
    gradient, and must have the same first dimension as indices, i.e., the nnz
    represented by indices and values must be consistent. Shape, if not empty or
    None, must be consistent with the accumulator's shape (if also provided).
    Example:
      A tensor [[0, 0], [0, 1], [2, 3]] can be represented
        indices: [1,2]
        values: [[0,1],[2,3]]
        shape: [3, 2]
    Args:
      grad_indices: Indices of the sparse gradient to be applied.
      grad_values: Values of the sparse gradient to be applied.
      grad_shape: Shape of the sparse gradient to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.
    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.
    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
    return gen_data_flow_ops.sparse_accumulator_apply_gradient(
        self._accumulator_ref,
        local_step=local_step,
        gradient_indices=math_ops.cast(grad_indices, _dtypes.int64),
        gradient_values=grad_values,
        gradient_shape=math_ops.cast(
            [] if grad_shape is None else grad_shape, _dtypes.int64),
        has_known_shape=(grad_shape is not None),
        name=name)
  def take_grad(self, num_required, name=None):
    """Attempts to extract the average gradient from the accumulator.
    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.
    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.
    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation
    Returns:
      A tuple of indices, values, and shape representing the average gradient.
    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
    return gen_data_flow_ops.sparse_accumulator_take_gradient(
        self._accumulator_ref, num_required, dtype=self._dtype, name=name)
  def take_indexed_slices_grad(self, num_required, name=None):
    """Attempts to extract the average gradient from the accumulator.
    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.
    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.
    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation
    Returns:
      An `IndexedSlices` holding the value of the average gradient.
    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
    return_val = gen_data_flow_ops.sparse_accumulator_take_gradient(
        self._accumulator_ref, num_required, dtype=self._dtype, name=name)
    return indexed_slices.IndexedSlices(
        indices=return_val.indices,
        values=return_val.values,
        dense_shape=return_val.shape)
  # SparseConditionalAccumulator is not switched to resource. Use old kernels.
  def num_accumulated(self, name=None):
    """Number of gradients that have currently been aggregated in accumulator.
    Args:
      name: Optional name for the operation.
    Returns:
      Number of accumulated gradients currently in accumulator.
    """
    if name is None:
      name = "%s_NumAccumulated" % self._name
    return gen_data_flow_ops.accumulator_num_accumulated(
        self._accumulator_ref, name=name)
  def set_global_step(self, new_global_step, name=None):
    """Sets the global time step of the accumulator.
    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.
    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.
    Returns:
      Operation that sets the accumulator's time step.
    """
    return gen_data_flow_ops.accumulator_set_global_step(
        self._accumulator_ref,
        math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64),
        name=name)
class BaseStagingArea:
  """Base class for Staging Areas."""
  _identifier = 0
  _lock = threading.Lock()
  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               capacity=0,
               memory_limit=0):
    if shared_name is None:
      self._name = (
          ops.get_default_graph().unique_name(self.__class__.__name__))
    elif isinstance(shared_name, str):
      self._name = shared_name
    else:
      raise ValueError(f"shared_name must be a string, got {shared_name}")
    self._dtypes = dtypes
    if shapes is not None:
      if len(shapes) != len(dtypes):
        raise ValueError("StagingArea shapes must be the same length as dtypes")
      self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
    else:
      self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
    if names is not None:
      if len(names) != len(dtypes):
        raise ValueError("StagingArea names must be the same length as dtypes")
      self._names = names
    else:
      self._names = None
    self._capacity = capacity
    self._memory_limit = memory_limit
    # all get and put ops must colocate with this op
    with ops.name_scope("%s_root" % self._name):
      self._coloc_op = control_flow_ops.no_op()
  @property
  def name(self):
    """The name of the staging area."""
    return self._name
  @property
  def dtypes(self):
    """The list of dtypes for each component of a staging area element."""
    return self._dtypes
  @property
  def shapes(self):
    """The list of shapes for each component of a staging area element."""
    return self._shapes
  @property
  def names(self):
    """The list of names for each component of a staging area element."""
    return self._names
  @property
  def capacity(self):
    """The maximum number of elements of this staging area."""
    return self._capacity
  @property
  def memory_limit(self):
    """The maximum number of bytes of this staging area."""
    return self._memory_limit
  def _check_put_dtypes(self, vals, indices=None):
    """Validate and convert `vals` to a list of `Tensor`s.
    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.
    If `vals` is a list, then the appropriate indices associated with the
    values must be provided.
    If it is a dictionary, the staging area must have been constructed with a
    `names` attribute and the dictionary keys must match the staging area names.
    `indices` will be inferred from the dictionary keys.
    If the staging area was constructed with a `names` attribute, `vals` must
    be a dictionary.
    Checks that the dtype and shape of each value matches that
    of the staging area.
    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.
    Returns:
      A (tensors, indices) tuple where `tensors` is a list of `Tensor` objects
      and `indices` is a list of indices associated with the tensors.
    Raises:
      ValueError: If `vals` or `indices` is invalid.
    """
    if isinstance(vals, dict):
      if not self._names:
        raise ValueError(
            "Staging areas must have names to enqueue a dictionary")
      if not set(vals.keys()).issubset(self._names):
        raise ValueError("Keys in dictionary to put do not match names "
                         f"of staging area. Dictionary: {sorted(vals.keys())}"
                         f"Queue: {sorted(self._names)}")
      # The order of values in `self._names` indicates the order in which the
      # tensors in the dictionary `vals` must be listed.
      vals, indices, _ = zip(*[(vals[k], i, k)
                               for i, k in enumerate(self._names)
                               if k in vals])
    else:
      if self._names:
        raise ValueError("You must enqueue a dictionary in a staging area "
                         "with names")
      if indices is None:
        raise ValueError("Indices must be supplied when inserting a list "
                         "of tensors")
      if len(indices) != len(vals):
        raise ValueError(f"Number of indices {len(indices)} doesn't match "
                         f"number of values {len(vals)}")
      if not isinstance(vals, (list, tuple)):
        vals = [vals]
        indices = [0]
    # Sanity check number of values
    if not len(vals) <= len(self._dtypes):
      raise ValueError(f"Unexpected number of inputs {len(vals)} vs "
                       f"{len(self._dtypes)}")
    tensors = []
    for val, i in zip(vals, indices):
      dtype, shape = self._dtypes[i], self._shapes[i]
      # Check dtype
      if val.dtype != dtype:
        raise ValueError(f"Datatypes do not match. "
                         f"Received val.dtype {str(val.dtype)} and "
                         f"dtype {str(dtype)}")
      # Check shape
      val.get_shape().assert_is_compatible_with(shape)
      tensors.append(
          ops.convert_to_tensor(val, dtype=dtype, name="component_%d" % i))
    return tensors, indices
  def _create_device_transfers(self, tensors):
    """Encode inter-device transfers if the current device
    is not the same as the Staging Area's device.
    """
    if not isinstance(tensors, (tuple, list)):
      tensors = [tensors]
    curr_device_scope = control_flow_ops.no_op().device
    if curr_device_scope != self._coloc_op.device:
      tensors = [array_ops.identity(t) for t in tensors]
    return tensors
  def _get_return_value(self, tensors, indices):
    """Return the value to return from a get op.
    If the staging area has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.
    Args:
      tensors: List of tensors from the get op.
      indices: Indices of associated names and shapes
    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """
    tensors = self._create_device_transfers(tensors)
    # Sets shape
    for output, i in zip(tensors, indices):
      output.set_shape(self._shapes[i])
    if self._names:
      # The returned values in `tensors` are in the same order as
      # the names in `self._names`.
      return {self._names[i]: t for t, i in zip(tensors, indices)}
    return tensors
  def _scope_vals(self, vals):
    """Return a list of values to pass to `name_scope()`.
    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.
    Returns:
      The values in vals as a list.
    """
    if isinstance(vals, (list, tuple)):
      return vals
    elif isinstance(vals, dict):
      return vals.values()
    else:
      return [vals]
class StagingArea(BaseStagingArea):
  """Class for staging inputs. No ordering guarantees.
  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.
  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.
  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.
  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.
  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,
  It can be configured with a capacity in which case
  put(values) will block until space becomes available.
  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.
  All get() and peek() commands block if the requested data
  is not present in the Staging Area.
  """
  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               capacity=0,
               memory_limit=0):
    """Constructs a staging area object.
    The two optional lists, `shapes` and `names`, must be of the same length
    as `dtypes` if provided.  The values at a given index `i` indicate the
    shape and name to use for the corresponding queue component in `dtypes`.
    The device scope at the time of object creation determines where the
    storage for the `StagingArea` will reside.  Calls to `put` will incur a copy
    to this memory space, if necessary.  Tensors returned by `get` will be
    placed according to the device scope when `get` is called.
    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area.
        An integer. If zero, the Staging Area is unbounded
    Raises:
      ValueError: If one of the arguments is invalid.
    """
    super(StagingArea, self).__init__(dtypes, shapes, names, shared_name,
                                      capacity, memory_limit)
  def put(self, values, name=None):
    """Create an op that places a value into the staging area.
    This operation will block if the `StagingArea` has reached
    its capacity.
    Args:
      values: A single tensor, a list or tuple of tensors, or a dictionary with
        tensor values. The number of elements must match the length of the
        list provided to the dtypes argument when creating the StagingArea.
      name: A name for the operation (optional).
    Returns:
        The created op.
    Raises:
      ValueError: If the number or type of inputs don't match the staging area.
    """
    with ops.name_scope(name, "%s_put" % self._name,
                        self._scope_vals(values)) as scope:
      if not isinstance(values, (list, tuple, dict)):
        values = [values]
      # Hard-code indices for this staging area
      indices = list(range(len(values)))
      vals, _ = self._check_put_dtypes(values, indices)
      with ops.colocate_with(self._coloc_op):
        op = gen_data_flow_ops.stage(
            values=vals,
            shared_name=self._name,
            name=scope,
            capacity=self._capacity,
            memory_limit=self._memory_limit)
      return op
  def __internal_get(self, get_fn, name):
    with ops.colocate_with(self._coloc_op):
      ret = get_fn()
    indices = list(range(len(self._dtypes)))  # Hard coded
    return self._get_return_value(ret, indices)
  def get(self, name=None):
    """Gets one element from this staging area.
    If the staging area is empty when this operation executes, it will block
    until there is an element to dequeue.
    Note that unlike others ops that can block, like the queue Dequeue
    operations, this can stop other work from happening.  To avoid this, the
    intended use is for this to be called only when there will be an element
    already available.  One method for doing this in a training loop would be to
    run a `put()` call during a warmup session.run call, and then call both
    `get()` and `put()` in each subsequent step.
    The placement of the returned tensor will be determined by the current
    device scope when this function is called.
    Args:
      name: A name for the operation (optional).
    Returns:
      The tuple of tensors that was gotten.
    """
    if name is None:
      name = "%s_get" % self._name
    # pylint: disable=bad-continuation
    fn = lambda: gen_data_flow_ops.unstage(dtypes=self._dtypes,
                    shared_name=self._name, name=name,
                    capacity=self._capacity,
                    memory_limit=self._memory_limit)
    # pylint: enable=bad-continuation
    return self.__internal_get(fn, name)
  def peek(self, index, name=None):
    """Peeks at an element in the staging area.
    If the staging area is too small to contain the element at
    the specified index, it will block until enough elements
    are inserted to complete the operation.
    The placement of the returned tensor will be determined by
    the current device scope when this function is called.
    Args:
      index: The index of the tensor within the staging area
              to look up.
      name: A name for the operation (optional).
    Returns:
      The tuple of tensors that was gotten.
    """
    if name is None:
      name = "%s_peek" % self._name
    # pylint: disable=bad-continuation
    fn = lambda: gen_data_flow_ops.stage_peek(index,
                    dtypes=self._dtypes, shared_name=self._name,
                    name=name, capacity=self._capacity,
                    memory_limit=self._memory_limit)
    # pylint: enable=bad-continuation
    return self.__internal_get(fn, name)
  def size(self, name=None):
    """Returns the number of elements in the staging area.
    Args:
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_size" % self._name
    return gen_data_flow_ops.stage_size(
        name=name,
        shared_name=self._name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)
  def clear(self, name=None):
    """Clears the staging area.
    Args:
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_clear" % self._name
    return gen_data_flow_ops.stage_clear(
        name=name,
        shared_name=self._name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)
class MapStagingArea(BaseStagingArea):
  """A `MapStagingArea` is a TensorFlow data structure that stores tensors
  across multiple steps, and exposes operations that can put and get tensors.
  Each `MapStagingArea` element is a (key, value) pair.
  Only int64 keys are supported, other types should be
  hashed to produce a key.
  Values are a tuple of one or more tensors.
  Each tuple component has a static dtype,
  and may have a static shape.
  The capacity of a `MapStagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.
  Each value tuple of a `MapStagingArea` is a fixed-length tuple of tensors
  whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.
  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,
  It behaves like an associative container with support for:
   - put(key, values)
   - peek(key)         like dict.get(key)
   - get(key)          like dict.pop(key)
   - get(key=None)     like dict.popitem()
   - size()
   - clear()
  If ordered a tree structure ordered by key will be used and
  get(key=None) will remove (key, value) pairs in increasing key order.
  Otherwise a hashtable
  It can be configured with a capacity in which case
  put(key, values) will block until space becomes available.
  Similarly, it can be configured with a memory limit which
  will block put(key, values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.
  All get() and peek() commands block if the requested
  (key, value) pair is not present in the staging area.
  Partial puts are supported and will be placed in an incomplete
  map until such time as all values associated with the key have
  been inserted. Once completed, this (key, value) pair will be
  inserted into the map. Data in the incomplete map
  counts towards the memory limit, but not towards capacity limit.
  Partial gets from the map are also supported.
  This removes the partially requested tensors from the entry,
  but the entry is only removed from the map once all tensors
  associated with it are removed.
  """
  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               ordered=False,
               capacity=0,
               memory_limit=0):
    """Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area (excluding keys).
        An integer. If zero, the Staging Area is unbounded
      ordered: (Optional.) If True the underlying data structure
        is a tree ordered on key. Otherwise assume a hashtable.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.
    Raises:
      ValueError: If one of the arguments is invalid.
    """
    super(MapStagingArea, self).__init__(dtypes, shapes, names, shared_name,
                                         capacity, memory_limit)
    # Defer to different methods depending if the map is ordered
    self._ordered = ordered
    if ordered:
      self._put_fn = gen_data_flow_ops.ordered_map_stage
      self._pop_fn = gen_data_flow_ops.ordered_map_unstage
      self._popitem_fn = gen_data_flow_ops.ordered_map_unstage_no_key
      self._peek_fn = gen_data_flow_ops.ordered_map_peek
      self._size_fn = gen_data_flow_ops.ordered_map_size
      self._incomplete_size_fn = gen_data_flow_ops.ordered_map_incomplete_size
      self._clear_fn = gen_data_flow_ops.ordered_map_clear
    else:
      self._put_fn = gen_data_flow_ops.map_stage
      self._pop_fn = gen_data_flow_ops.map_unstage
      self._popitem_fn = gen_data_flow_ops.map_unstage_no_key
      self._peek_fn = gen_data_flow_ops.map_peek
      self._size_fn = gen_data_flow_ops.map_size
      self._incomplete_size_fn = gen_data_flow_ops.map_incomplete_size
      self._clear_fn = gen_data_flow_ops.map_clear
  def put(self, key, vals, indices=None, name=None):
    """Create an op that stores the (key, vals) pair in the staging area.
    Incomplete puts are possible, preferably using a dictionary for vals
    as the appropriate dtypes and shapes can be inferred from the value names
    dictionary key values. If vals is a list or tuple, indices must
    also be specified so that the op knows at which element position
    to perform the insert.
    This operation will block if the capacity or memory limit of this
    container is reached.
    Args:
        key: Key associated with the data
        vals: Tensor (or a dict/tuple of Tensors) to place
                into the staging area.
        indices: (Optional) if vals is a tuple/list, this is required.
        name: A name for the operation (optional)
    Returns:
        The created op
    Raises:
        ValueError: If the number or type of inputs don't match the staging
        area.
    """
    with ops.name_scope(name, "%s_put" % self._name,
                        self._scope_vals(vals)) as scope:
      vals, indices = self._check_put_dtypes(vals, indices)
      with ops.colocate_with(self._coloc_op):
        op = self._put_fn(
            key,
            indices,
            vals,
            dtypes=self._dtypes,
            shared_name=self._name,
            name=scope,
            capacity=self._capacity,
            memory_limit=self._memory_limit)
    return op
  def _get_indices_and_dtypes(self, indices=None):
    if indices is None:
      indices = list(range(len(self._dtypes)))
    if not isinstance(indices, (tuple, list)):
      raise TypeError(f"Invalid indices type {type(indices)}")
    if len(indices) == 0:
      raise ValueError("Empty indices")
    if all(isinstance(i, str) for i in indices):
      if self._names is None:
        raise ValueError(f"String indices provided {indices}, but "
                         "this Staging Area was not created with names.")
      try:
        indices = [self._names.index(n) for n in indices]
      except ValueError:
        raise ValueError(f"Named index not in "
                         f"Staging Area names {self._names}")
    elif all(isinstance(i, int) for i in indices):
      pass
    else:
      raise TypeError(f"Mixed types in indices {indices}. "
                      "May only be str or int")
    dtypes = [self._dtypes[i] for i in indices]
    return indices, dtypes
  def peek(self, key, indices=None, name=None):
    """Peeks at staging area data associated with the key.
    If the key is not in the staging area, it will block
    until the associated (key, value) is inserted.
    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_pop" % self._name
    indices, dtypes = self._get_indices_and_dtypes(indices)
    with ops.colocate_with(self._coloc_op):
      result = self._peek_fn(
          key,
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)
    return self._get_return_value(result, indices)
  def get(self, key=None, indices=None, name=None):
    """If the key is provided, the associated (key, value) is returned from the staging area.
    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    If no key is provided and the staging area is ordered,
    the (key, value) with the smallest key will be returned.
    Otherwise, a random (key, value) will be returned.
    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.
    Args:
        key: Key associated with the required data (Optional)
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if key is None:
      return self._popitem(indices=indices, name=name)
    else:
      return self._pop(key, indices=indices, name=name)
  def _pop(self, key, indices=None, name=None):
    """Remove and return the associated (key, value) is returned from the staging area.
    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_get" % self._name
    indices, dtypes = self._get_indices_and_dtypes(indices)
    with ops.colocate_with(self._coloc_op):
      result = self._pop_fn(
          key,
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)
    return key, self._get_return_value(result, indices)
  def _popitem(self, indices=None, name=None):
    """If the staging area is ordered, the (key, value) with the smallest key will be returned.
    Otherwise, a random (key, value) will be returned.
    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.
    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_get_nokey" % self._name
    indices, dtypes = self._get_indices_and_dtypes(indices)
    with ops.colocate_with(self._coloc_op):
      key, result = self._popitem_fn(
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)
    # Separate keys and results out from
    # underlying namedtuple
    key = self._create_device_transfers(key)[0]
    result = self._get_return_value(result, indices)
    return key, result
  def size(self, name=None):
    """Returns the number of elements in the staging area.
    Args:
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_size" % self._name
    return self._size_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)
  def incomplete_size(self, name=None):
    """Returns the number of incomplete elements in the staging area.
    Args:
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_incomplete_size" % self._name
    return self._incomplete_size_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)
  def clear(self, name=None):
    """Clears the staging area.
    Args:
        name: A name for the operation (optional)
    Returns:
        The created op
    """
    if name is None:
      name = "%s_clear" % self._name
    return self._clear_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)
class RecordInput:
  """RecordInput asynchronously reads and randomly yields TFRecords.
  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.
  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.
  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  """
  def __init__(self,
               file_pattern,
               batch_size=1,
               buffer_size=1,
               parallelism=1,
               shift_ratio=0,
               seed=0,
               name=None,
               batches=None,
               compression_type=None):
    """Constructs a RecordInput Op.
    Args:
      file_pattern: File path to the dataset, possibly containing wildcards.
        All matching files will be iterated over each epoch.
      batch_size: How many records to return at a time.
      buffer_size: The maximum number of records the buffer will contain.
      parallelism: How many reader threads to use for reading from files.
      shift_ratio: What percentage of the total number files to move the start
        file forward by each epoch.
      seed: Specify the random number seed used by generator that randomizes
        records.
      name: Optional name for the operation.
      batches: None by default, creating a single batch op. Otherwise specifies
        how many batches to create, which are returned as a list when
        `get_yield_op()` is called. An example use case is to split processing
        between devices on one computer.
      compression_type: The type of compression for the file. Currently ZLIB and
        GZIP are supported. Defaults to none.
    Raises:
      ValueError: If one of the arguments is invalid.
    """
    self._batch_size = batch_size
    if batches is not None:
      self._batch_size *= batches
    self._batches = batches
    self._file_pattern = file_pattern
    self._buffer_size = buffer_size
    self._parallelism = parallelism
    self._shift_ratio = shift_ratio
    self._seed = seed
    self._name = name
    self._compression_type = python_io.TFRecordCompressionType.NONE
    if compression_type is not None:
      self._compression_type = compression_type
  def get_yield_op(self):
    """Adds a node that yields a group of records every time it is executed.
    If RecordInput `batches` parameter is not None, it yields a list of
    record batches with the specified `batch_size`.
    """
    compression_type = python_io.TFRecordOptions.get_compression_type_string(
        python_io.TFRecordOptions(self._compression_type))
    records = gen_data_flow_ops.record_input(
        file_pattern=self._file_pattern,
        file_buffer_size=self._buffer_size,
        file_parallelism=self._parallelism,
        file_shuffle_shift_ratio=self._shift_ratio,
        batch_size=self._batch_size,
        file_random_seed=self._seed,
        compression_type=compression_type,
        name=self._name)
    if self._batches is None:
      return records
    else:
      with ops.name_scope(self._name):
        batch_list = [[] for _ in range(self._batches)]
        records = array_ops.split(records, self._batch_size, 0)
        for index, protobuf in enumerate(records):
          batch_index = index % self._batches
          batch_list[batch_index].append(array_ops.reshape(protobuf, []))
        return batch_list
