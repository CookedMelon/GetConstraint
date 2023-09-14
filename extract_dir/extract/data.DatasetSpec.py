@tf_export(
    "data.DatasetSpec",
    v1=["data.DatasetSpec", "data.experimental.DatasetStructure"])
class DatasetSpec(type_spec.BatchableTypeSpec):
  """Type specification for `tf.data.Dataset`.
  See `tf.TypeSpec` for more information about TensorFlow type specifications.
  >>> dataset = tf.data.Dataset.range(3)
  >>> tf.data.DatasetSpec.from_value(dataset)
  DatasetSpec(TensorSpec(shape=(), dtype=tf.int64, name=None), TensorShape([]))
  """
  __slots__ = ["_element_spec", "_dataset_shape"]
  def __init__(self, element_spec, dataset_shape=()):
    self._element_spec = element_spec
    self._dataset_shape = tensor_shape.as_shape(dataset_shape)
  @property
  def value_type(self):
    return Dataset
  @property
  def element_spec(self):
    """The inner element spec."""
    return self._element_spec
  def is_subtype_of(self, other):
    """See base class."""
    if type(self) is not type(other):
      return False
    # TODO(b/220385675): _element_spec should always be a TypeSpec.
    try:
      tf_nest.assert_same_structure(self.element_spec, other.element_spec)
    except (TypeError, ValueError):
      return False
    self_elements = tf_nest.flatten(self.element_spec)
    other_elements = tf_nest.flatten(other.element_spec)
    def is_subtype_or_equal(a, b):
      if isinstance(a, trace.TraceType):
        return a.is_subtype_of(b)
      else:
        return a == b
    for self_element, other_element in zip(self_elements, other_elements):
      if not is_subtype_or_equal(self_element, other_element):
        return False
    return self._dataset_shape.is_subtype_of(other._dataset_shape)  # pylint: disable=protected-access
  def most_specific_common_supertype(self, others):
    """See base class."""
    if not all(type(self) is type(other) for other in others):
      return None
    try:
      for other in others:
        tf_nest.assert_same_structure(self.element_spec, other.element_spec)
    except (TypeError, ValueError):
      return None
    self_components = tf_nest.flatten(self.element_spec)
    others_components = [
        tf_nest.flatten(other.element_spec) for other in others
    ]
    common_components = [None] * len(self_components)
    def common_supertype_or_equal(a, bs):
      if isinstance(a, trace.TraceType):
        return a.most_specific_common_supertype(bs)
      else:
        return a if all(a == b for b in bs) else None
    for i, self_component in enumerate(self_components):
      common_components[i] = common_supertype_or_equal(
          self_component,
          [other_components[i] for other_components in others_components])
      if self_component is not None and common_components[i] is None:
        return None
    common_element_spec = tf_nest.pack_sequence_as(self._element_spec,
                                                   common_components)
    common_dataset_shape = self._dataset_shape.most_specific_common_supertype(
        [other._dataset_shape for other in others])  # pylint: disable=protected-access
    if common_dataset_shape is None:
      return None
    return DatasetSpec(common_element_spec, common_dataset_shape)
  # TODO(b/220385675): Once _element_spec is guaranteed to be TypeSpec, the
  # following functions do not need to be overloaded: is_subtype_of,
  # most_specific_common_supertype, __hash__ and __eq__
  def _serialize(self):
    return (self._element_spec, self._dataset_shape)
  @property
  def _component_specs(self):
    return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)
  def _to_components(self, value):
    return value._variant_tensor  # pylint: disable=protected-access
  def _from_components(self, components):
    # pylint: disable=protected-access
    if self._dataset_shape.ndims == 0:
      return _VariantDataset(components, self._element_spec)
    else:
      return _NestedVariant(components, self._element_spec, self._dataset_shape)
  def _to_tensor_list(self, value):
    return [
        ops.convert_to_tensor(
            tf_nest.map_structure(lambda x: x._variant_tensor, value))  # pylint: disable=protected-access
    ]
  @staticmethod
  def from_value(value):
    """Creates a `DatasetSpec` for the given `tf.data.Dataset` value."""
    return DatasetSpec(value.element_spec)  # pylint: disable=protected-access
  def _batch(self, batch_size):
    return DatasetSpec(
        self._element_spec,
        tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))
  def _unbatch(self):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Slicing dataset elements is not supported for rank 0.")
    return DatasetSpec(self._element_spec, self._dataset_shape[1:])
  def _to_batched_tensor_list(self, value):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Slicing dataset elements is not supported for rank 0.")
    return self._to_tensor_list(value)
  def _to_legacy_output_types(self):
    return self
  def _to_legacy_output_shapes(self):
    return self
  def _to_legacy_output_classes(self):
    return self
  def __hash__(self):
    # TODO(b/220385675): attributes can be dicts and hence unhashable.
    return hash(DatasetSpec)
  def __eq__(self, other):
    return (isinstance(other, DatasetSpec) and
            self._element_spec == other._element_spec and
            self._dataset_shape == other._dataset_shape)
nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        DatasetSpec, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC
    )
)
class _NumpyIterator(tracking_base.Trackable):
  """Iterator over a dataset with elements converted to numpy."""
  __slots__ = ["_iterator"]
  def __init__(self, dataset):
    self._iterator = iter(dataset)
  def __iter__(self):
    return self
  def __next__(self):
    def to_numpy(x):
      numpy = x._numpy()  # pylint: disable=protected-access
      if isinstance(numpy, np.ndarray):
        # `numpy` shares the same underlying buffer as the `x` Tensor.
        # Tensors are expected to be immutable, so we disable writes.
        numpy.setflags(write=False)
      return numpy
    return nest.map_structure(to_numpy, next(self._iterator))
  def next(self):
    return self.__next__()
  # override
  def _serialize_to_tensors(self):
    # pylint: disable=protected-access
    return self._iterator._serialize_to_tensors()
  # override
  def _restore_from_tensors(self, restored_tensors):
    # pylint: disable=protected-access
    return self._iterator._restore_from_tensors(restored_tensors)
  def _save(self):
    # pylint: disable=protected-access
    return self._iterator._save()
  def _restore(self, state):
    # pylint: disable=protected-access
    return self._iterator._restore(state)
class _VariantTracker(resource_lib.CapturableResource):
  """Allows export of functions capturing a Dataset in SavedModels.
  When saving a SavedModel, `tf.saved_model.save` traverses the object
  graph. Since Datasets reference _VariantTracker objects, that traversal will
  find a _VariantTracker for each Dataset and so know how to save and restore
  functions which reference the Dataset's variant Tensor.
  """
  def __init__(self, variant_tensor, resource_creator):
    """Record that `variant_tensor` is associated with `resource_creator`.
    Args:
      variant_tensor: The variant-dtype Tensor associated with the Dataset. This
        Tensor will be a captured input to functions which use the Dataset, and
        is used by saving code to identify the corresponding _VariantTracker.
      resource_creator: A zero-argument function which creates a new
        variant-dtype Tensor. This function will be included in SavedModels and
        run to re-create the Dataset's variant Tensor on restore.
    """
    super(_VariantTracker, self).__init__(device="CPU")
    self._resource_handle = variant_tensor
    if not isinstance(resource_creator, def_function.Function):
      # Internal validation -- _VariantTracker assumes that resource creator is
      # already a tf.function.
      raise TypeError("Resource creator should already be a tf.function.")
    self._create_resource = resource_creator
  def _trackable_children(self,
                          save_type=tracking_base.SaveType.CHECKPOINT,
                          **kwargs):
    if save_type != tracking_base.SaveType.SAVEDMODEL:
      return {}
    children = super(_VariantTracker,
                     self)._trackable_children(save_type, **kwargs)
    # Overwrite the _create_resource function, since `self._create_resource`
    # is already a tf.function.
    children["_create_resource"] = self._create_resource
    return children
# TODO(b/254291122): Remove.
# Loaded lazily due to a circular dependency (dataset_ops ->
# batch_op -> dataset_ops).
batch_op = lazy_loader.LazyLoader(
    "batch_op", globals(),
    "tensorflow.python.data.ops.batch_op")
BatchDataset = batch_op._BatchDataset  # pylint: disable=protected-access
PrefetchDataset = prefetch_op._PrefetchDataset  # pylint: disable=protected-access
ShuffleDataset = shuffle_op._ShuffleDataset  # pylint: disable=protected-access
# TODO(b/254291122): Remove.
# Loaded lazily due to a circular dependency (dataset_ops ->
# repeat_op -> dataset_ops).
repeat_op = lazy_loader.LazyLoader(
    "repeat_op", globals(),
    "tensorflow.python.data.ops.repeat_op")
RepeatDataset = repeat_op._RepeatDataset  # pylint: disable=protected-access
class _OptionsDataset(UnaryUnchangedStructureDataset):
  """An identity `Dataset` that stores options."""
  def __init__(self, input_dataset, options, name=None):
    # pylint: disable=protected-access
    self._input_dataset = input_dataset
    options_pb = dataset_options_pb2.Options()
    options_pb.CopyFrom(options._to_proto())
    self._name = name
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.options_dataset(
          input_dataset._variant_tensor, options_pb.SerializeToString(),
          **self._common_args)
    super(_OptionsDataset, self).__init__(input_dataset, variant_tensor)
    if self._options_attr:
      self._options_attr._set_mutable(True)
      self._options_attr = self._options_attr.merge(options)
    else:
      self._options_attr = options
    self._options_attr._set_mutable(False)
def normalize_to_dense(dataset):
  """Normalizes non-tensor components in a dataset to dense representations.
  This is necessary for dataset transformations that slice along the batch
  dimension and are oblivious to non-tensors, e.g. `unbatch`, `rebatch`.
  Args:
    dataset: Dataset to normalize.
  Returns:
    A dataset whose sparse and ragged tensors have been normalized to their
    dense representations.
  """
  # NOTE(mrry): This leads to a somewhat inefficient re-encoding step for all
  # non-tensor components.
  #
  # TODO(mrry): Consider optimizing this if it turns out to be a bottleneck.
  if structured_function._should_unpack(dataset.element_spec):  # pylint: disable=protected-access
    def normalize(*args):
      return structure.to_batched_tensor_list(dataset.element_spec, tuple(args))
  else:
    def normalize(arg):
      return structure.to_batched_tensor_list(dataset.element_spec, arg)
  normalized_dataset = dataset.map(normalize)
  # NOTE(mrry): Our `map()` has lost information about the structure of
  # non-tensor components, so re-apply the structure of the original dataset.
  return _RestructuredDataset(normalized_dataset, dataset.element_spec)
class _RestructuredDataset(UnaryDataset):
  """An internal helper for changing the element spec of a dataset."""
  def __init__(self, dataset, element_spec):
    self._input_dataset = dataset
    self._element_spec = element_spec
    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_RestructuredDataset, self).__init__(dataset, variant_tensor)
  @property
  def element_spec(self):
    return self._element_spec
def _get_prob_original_static(initial_dist_t, target_dist_t):
  """Returns the static probability of sampling from the original.
  `tensor_util.constant_value(prob_of_original)` returns `None` if it encounters
  an Op that it isn't defined for. We have some custom logic to avoid this.
  Args:
    initial_dist_t: A tensor of the initial distribution.
    target_dist_t: A tensor of the target distribution.
  Returns:
    The probability of sampling from the original distribution as a constant,
    if it is a constant, or `None`.
  """
  init_static = tensor_util.constant_value(initial_dist_t)
  target_static = tensor_util.constant_value(target_dist_t)
  if init_static is None or target_static is None:
    return None
  else:
    return np.min(target_static / init_static)
def _filter_ds(dataset,
               acceptance_dist_ds,
               initial_dist_ds,
               class_func,
               seed,
               name=None):
  """Filters a dataset based on per-class acceptance probabilities.
  Args:
    dataset: The dataset to be filtered.
    acceptance_dist_ds: A dataset of acceptance probabilities.
    initial_dist_ds: A dataset of the initial probability distribution, given or
      estimated.
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    seed: (Optional.) Python integer seed for the resampler.
    name: (Optional.) A name for the tf.data operation.
  Returns:
    A dataset of (class value, data) after filtering.
  """
  def maybe_warn_on_large_rejection(accept_dist, initial_dist):
    proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
    return cond.cond(
        math_ops.less(proportion_rejected, .5),
        lambda: accept_dist,
        lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
            accept_dist, [proportion_rejected, initial_dist, accept_dist],
            message="Proportion of examples rejected by sampler is high: ",
            summarize=100,
            first_n=10))
  acceptance_dist_ds = (
      DatasetV2.zip((acceptance_dist_ds, initial_dist_ds),
                    name=name).map(maybe_warn_on_large_rejection, name=name))
  def _gather_and_copy(acceptance_prob, data):
    if isinstance(data, tuple):
      class_val = class_func(*data)
    else:
      class_val = class_func(data)
    return class_val, array_ops.gather(acceptance_prob, class_val), data
  current_probabilities_and_class_and_data_ds = DatasetV2.zip(
      (acceptance_dist_ds, dataset), name=name).map(
          _gather_and_copy, name=name)
  def _reject(unused_class_val, p, unused_data):
    return random_ops.random_uniform([], seed=seed, dtype=p.dtype) < p
  filtered_ds = current_probabilities_and_class_and_data_ds.filter(
      _reject, name=name)
  return filtered_ds.map(
      lambda class_value, _, data: (class_value, data), name=name)
# pylint: disable=missing-function-docstring
def _estimate_initial_dist_ds(target_dist_t,
                              class_values_ds,
                              dist_estimation_batch_size=32,
                              smoothing_constant=10,
                              name=None):
  num_classes = (target_dist_t.shape[0] or array_ops.shape(target_dist_t)[0])
  initial_examples_per_class_seen = array_ops.fill([num_classes],
                                                   np.int64(smoothing_constant))
  def update_estimate_and_tile(num_examples_per_class_seen, c):
    updated_examples_per_class_seen, dist = _estimate_data_distribution(
        c, num_examples_per_class_seen)
    tiled_dist = array_ops.tile(
        array_ops.expand_dims(dist, 0), [dist_estimation_batch_size, 1])
    return updated_examples_per_class_seen, tiled_dist
  initial_dist_ds = (
      class_values_ds.batch(dist_estimation_batch_size, name=name).scan(
          initial_examples_per_class_seen, update_estimate_and_tile,
          name=name).unbatch(name=name))
  return initial_dist_ds
def _get_target_to_initial_ratio(initial_probs, target_probs):
  # Add tiny to initial_probs to avoid divide by zero.
  denom = (initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny)
  return target_probs / denom
def _estimate_data_distribution(c, num_examples_per_class_seen):
  """Estimate data distribution as labels are seen.
  Args:
    c: The class labels.  Type `int32`, shape `[batch_size]`.
    num_examples_per_class_seen: Type `int64`, shape `[num_classes]`, containing
      counts.
  Returns:
    num_examples_per_lass_seen: Updated counts.  Type `int64`, shape
      `[num_classes]`.
    dist: The updated distribution.  Type `float32`, shape `[num_classes]`.
  """
  num_classes = num_examples_per_class_seen.get_shape()[0]
  # Update the class-count based on what labels are seen in batch.
  num_examples_per_class_seen = math_ops.add(
      num_examples_per_class_seen,
      math_ops.reduce_sum(
          array_ops.one_hot(c, num_classes, dtype=dtypes.int64), 0))
  init_prob_estimate = math_ops.truediv(
      num_examples_per_class_seen,
      math_ops.reduce_sum(num_examples_per_class_seen))
  dist = math_ops.cast(init_prob_estimate, dtypes.float32)
  return num_examples_per_class_seen, dist
def _calculate_acceptance_probs_with_mixing(initial_probs, target_probs):
  """Calculates the acceptance probabilities and mixing ratio.
  In this case, we assume that we can *either* sample from the original data
  distribution with probability `m`, or sample from a reshaped distribution
  that comes from rejection sampling on the original distribution. This
  rejection sampling is done on a per-class basis, with `a_i` representing the
  probability of accepting data from class `i`.
  This method is based on solving the following analysis for the reshaped
  distribution:
  Let F be the probability of a rejection (on any example).
  Let p_i be the proportion of examples in the data in class i (init_probs)
  Let a_i is the rate the rejection sampler should *accept* class i
  Let t_i is the target proportion in the minibatches for class i (target_probs)
  ```
  F = sum_i(p_i * (1-a_i))
    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1
  ```
  An example with class `i` will be accepted if `k` rejections occur, then an
  example with class `i` is seen by the rejector, and it is accepted. This can
  be written as follows:
  ```
  t_i = sum_k=0^inf(F^k * p_i * a_i)
      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1
      = p_i * a_i / sum_j(p_j * a_j)        using F from above
  ```
  Note that the following constraints hold:
  ```
  0 <= p_i <= 1, sum_i(p_i) = 1
  0 <= a_i <= 1
  0 <= t_i <= 1, sum_i(t_i) = 1
  ```
  A solution for a_i in terms of the other variables is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```
  If we try to minimize the amount of data rejected, we get the following:
  M_max = max_i [ t_i / p_i ]
  M_min = min_i [ t_i / p_i ]
  The desired probability of accepting data if it comes from class `i`:
  a_i = (t_i/p_i - m) / (M_max - m)
  The desired probability of pulling a data element from the original dataset,
  rather than the filtered one:
  m = M_min
  Args:
    initial_probs: A Tensor of the initial probability distribution, given or
      estimated.
    target_probs: A Tensor of the corresponding classes.
  Returns:
    (A 1D Tensor with the per-class acceptance probabilities, the desired
    probability of pull from the original distribution.)
  """
  ratio_l = _get_target_to_initial_ratio(initial_probs, target_probs)
  max_ratio = math_ops.reduce_max(ratio_l)
  min_ratio = math_ops.reduce_min(ratio_l)
  # Target prob to sample from original distribution.
  m = min_ratio
  # TODO(joelshor): Simplify fraction, if possible.
  a_i = (ratio_l - m) / (max_ratio - m)
  return a_i, m
def _apply_rewrite(dataset, rewrite):
  # pylint: disable=protected-access
  return _VariantDataset(
      gen_dataset_ops.rewrite_dataset(dataset._variant_tensor, rewrite,
                                      **dataset._flat_structure),
      dataset.element_spec)
def _collect_resource_inputs(op):
  """Collects resource inputs for the given ops (and its variant inputs)."""
  def _process(op_queue, seen_ops):
    """Processes the next element of the op queue.
    Args:
      op_queue: Queue of Dataset operations to process.
      seen_ops: Already processed set of Operations.
    Returns:
      A 2-tuple containing sets of resource handles. The first tuple entry
      contains read-only handles and the second entry contains read-write
      handles.
    """
    reads = []
    writes = []
    op = op_queue.pop()
    if op in seen_ops:
      return reads, writes
    seen_ops.add(op)
    # TODO(b/150139257): All resource inputs are in writes right now since we
    # have not updated the functional ops to set the special attribute that ACD
    # uses to figure out which of the op's inputs are read-only.
    reads, writes = acd_utils.get_read_write_resource_inputs(op)
    # Conservatively assume that any variant inputs are datasets.
    op_queue.extend(t.op for t in op.inputs if t.dtype == dtypes.variant)
    return reads, writes
  op_queue = [op]
  seen_ops = set()
  all_reads = []
  all_writes = []
  while op_queue:
    reads, writes = _process(op_queue, seen_ops)
    all_reads.extend(reads)
    all_writes.extend(writes)
  return all_reads, all_writes
@auto_control_deps.register_acd_resource_resolver
def _resource_resolver(op, resource_reads, resource_writes):
  """Updates resource inputs for tf.data ops with indirect dependencies."""
  updated = False
  if op.type in [
      "DatasetToSingleElement", "DatasetToTFRecord", "ReduceDataset"
  ]:
    reads, writes = _collect_resource_inputs(op)
    for inp in reads:
      if inp not in resource_reads:
        updated = True
        resource_reads.add(inp)
    for inp in writes:
      if inp not in resource_writes:
        updated = True
        resource_writes.add(inp)
  if op.type in [
      "IteratorGetNext", "IteratorGetNextSync", "IteratorGetNextAsOptional"
  ]:
    iterator_resource = op.inputs[0]
    make_iterator_ops = [
        op for op in iterator_resource.consumers() if op.type == "MakeIterator"
    ]
    if len(make_iterator_ops) == 1:
      reads, writes = _collect_resource_inputs(make_iterator_ops[0])
      for inp in reads:
        if inp not in resource_reads:
          updated = True
          resource_reads.add(inp)
      for inp in writes:
        if inp not in resource_writes:
          updated = True
          resource_writes.add(inp)
  return updated
dataset_autograph.register_overrides()
