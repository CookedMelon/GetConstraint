@tf_export("data.Dataset", v1=[])
class DatasetV2(
    collections_abc.Iterable,
    tracking_base.Trackable,
    composite_tensor.CompositeTensor,
    data_types.DatasetV2,
    metaclass=abc.ABCMeta):
  """Represents a potentially large set of elements.
  The `tf.data.Dataset` API supports writing descriptive and efficient input
  pipelines. `Dataset` usage follows a common pattern:
  1. Create a source dataset from your input data.
  2. Apply dataset transformations to preprocess the data.
  3. Iterate over the dataset and process the elements.
  Iteration happens in a streaming fashion, so the full dataset does not need to
  fit into memory.
  Source Datasets:
  The simplest way to create a dataset is to create it from a python `list`:
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)
  To process lines from files, use `tf.data.TextLineDataset`:
  >>> dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
  To process records written in the `TFRecord` format, use `TFRecordDataset`:
  >>> dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"])
  To create a dataset of all files matching a pattern, use
  `tf.data.Dataset.list_files`:
  ```python
  dataset = tf.data.Dataset.list_files("/path/*.txt")
  ```
  See `tf.data.FixedLengthRecordDataset` and `tf.data.Dataset.from_generator`
  for more ways to create datasets.
  Transformations:
  Once you have a dataset, you can apply transformations to prepare the data for
  your model:
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.map(lambda x: x*2)
  >>> list(dataset.as_numpy_iterator())
  [2, 4, 6]
  Common Terms:
  **Element**: A single output from calling `next()` on a dataset iterator.
    Elements may be nested structures containing multiple components. For
    example, the element `(1, (3, "apple"))` has one tuple nested in another
    tuple. The components are `1`, `3`, and `"apple"`.
  **Component**: The leaf in the nested structure of an element.
  Supported types:
  Elements can be nested structures of tuples, named tuples, and dictionaries.
  Note that Python lists are *not* treated as nested structures of components.
  Instead, lists are converted to tensors and treated as components. For
  example, the element `(1, [1, 2, 3])` has only two components; the tensor `1`
  and the tensor `[1, 2, 3]`. Element components can be of any type
  representable by `tf.TypeSpec`, including `tf.Tensor`, `tf.data.Dataset`,
  `tf.sparse.SparseTensor`, `tf.RaggedTensor`, and `tf.TensorArray`.
  ```python
  a = 1 # Integer element
  b = 2.0 # Float element
  c = (1, 2) # Tuple element with 2 components
  d = {"a": (2, 2), "b": 3} # Dict element with 3 components
  Point = collections.namedtuple("Point", ["x", "y"])
  e = Point(1, 2) # Named tuple
  f = tf.data.Dataset.range(10) # Dataset element
  ```
  For more information,
  read [this guide](https://www.tensorflow.org/guide/data).
  """
  def __init__(self, variant_tensor):
    """Creates a DatasetV2 object.
    This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not
    take anything in its constructor whereas in the DatasetV2, we expect
    subclasses to create a variant_tensor and pass it in to the super() call.
    Args:
      variant_tensor: A DT_VARIANT tensor that represents the dataset.
    """
    self._variant_tensor_attr = variant_tensor
    self._graph_attr = ops.get_default_graph()
    # Initialize the options for this dataset and its inputs.
    self._options_attr = options_lib.Options()
    for input_dataset in self._inputs():
      input_options = None
      if isinstance(input_dataset, data_types.DatasetV1):
        # If the V1 dataset does not have the `_dataset` attribute, we assume it
        # is a dataset source and hence does not have options. Otherwise, we
        # grab the options of `_dataset` object
        if hasattr(input_dataset, "_dataset"):
          if not isinstance(input_dataset._dataset, data_types.DatasetV2):
            raise TypeError(
                f"Each input of dataset {type(self)} should be a subclass of "
                f"`tf.data.Dataset` but encountered "
                f"{type(input_dataset._dataset)}.")
          input_options = input_dataset._dataset._options_attr
      elif isinstance(input_dataset, data_types.DatasetV2):
        input_options = input_dataset._options_attr
      else:
        raise TypeError(
            f"Each input of dataset {type(self)} should be a subclass of "
            f"`tf.data.Dataset` but encountered {type(input_dataset)}.")
      if input_options is not None:
        self._options_attr = self._options_attr.merge(input_options)
    self._options_attr._set_mutable(False)  # pylint: disable=protected-access
  @property
  def _variant_tensor(self):
    return self._variant_tensor_attr
  @_variant_tensor.setter
  def _variant_tensor(self, _):
    raise ValueError("The `_variant_tensor` property cannot be modified.")
  @deprecation.deprecated_args(None, "Use external_state_policy instead",
                               "allow_stateful")
  def _as_serialized_graph(
      self,
      allow_stateful=None,
      strip_device_assignment=None,
      external_state_policy=options_lib.ExternalStatePolicy.WARN):
    """Produces serialized graph representation of the dataset.
    Args:
      allow_stateful: If true, we allow stateful ops to be present in the graph
        def. In that case, the state in these ops would be thrown away.
      strip_device_assignment: If true, non-local (i.e. job and task) device
        assignment is stripped from ops in the serialized graph.
      external_state_policy: The ExternalStatePolicy enum that determines how we
        handle input pipelines that depend on external state. By default, its
        set to WARN.
    Returns:
      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a
      serialized graph.
    """
    if external_state_policy:
      policy = external_state_policy.value
      return gen_dataset_ops.dataset_to_graph_v2(
          self._variant_tensor,
          external_state_policy=policy,
          strip_device_assignment=strip_device_assignment)
    if strip_device_assignment:
      return gen_dataset_ops.dataset_to_graph(
          self._variant_tensor,
          allow_stateful=allow_stateful,
          strip_device_assignment=strip_device_assignment)
    return gen_dataset_ops.dataset_to_graph(
        self._variant_tensor, allow_stateful=allow_stateful)
  def _maybe_track_assets(self, graph_def):
    """Finds and tracks nodes in `graph_def` that refer to asset files.
    Args:
      graph_def: Serialized graph representation of this dataset.
    Returns:
      A dictionary mapping the node name of an asset constant to a tracked
      `asset.Asset` object.
    """
    asset_tracker = {}
    for node in graph_def.node:
      if node.name.startswith("FileIdentity"):
        asset_tracker[node.input[0]] = None
    if not asset_tracker:
      return {}
    for node in graph_def.node:
      if node.name in asset_tracker:
        tensor_proto = node.attr["value"].tensor
        with context.eager_mode(), ops.device("CPU"):
          node_value = gen_parsing_ops.parse_tensor(
              tensor_proto.SerializeToString(), dtypes.string).numpy()
        asset_tracker[node.name] = ([
            self._track_trackable(asset.Asset(n),
                                  name=node.name + "_" + str(i), overwrite=True)
            for i, n in enumerate(node_value)
        ])
    return asset_tracker
  def _trackable_children(self,
                          save_type=tracking_base.SaveType.CHECKPOINT,
                          **kwargs):
    if save_type != tracking_base.SaveType.SAVEDMODEL:
      return {}
    # _trace_variant_creation only works when executing eagerly, so we don't
    # want to run it in the object initialization.
    @def_function.function(input_signature=[], autograph=False)
    def _creator():
      resource = self._trace_variant_creation()()  # pylint: disable=protected-access
      return resource
    _creator.get_concrete_function()  # Trigger asset tracking
    children = super(DatasetV2, self)._trackable_children(save_type, **kwargs)
    children["_variant_tracker"] = _VariantTracker(self._variant_tensor,
                                                   _creator)
    return children
  def _trace_variant_creation(self):
    """Traces a function which outputs a variant `tf.Tensor` for this dataset.
    Note that creating this function involves evaluating an op, and is currently
    only supported when executing eagerly.
    Returns:
      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.
    """
    variant = self._variant_tensor
    if not isinstance(variant, ops.EagerTensor):
      raise NotImplementedError(
          "Constructing a tf.function that reproduces a given dataset is only "
          "supported for datasets created eagerly. Please file a feature "
          "request if this is important to you.")
    with context.eager_mode(), ops.device("CPU"):
      # pylint: disable=protected-access
      graph_def = graph_pb2.GraphDef().FromString(
          self._as_serialized_graph(external_state_policy=options_lib
                                    .ExternalStatePolicy.FAIL).numpy())
    output_node_names = []
    for node in graph_def.node:
      if node.op == "_Retval":
        output_node_names = node.input
    if len(output_node_names) != 1:
      raise AssertionError(
          f"Dataset graph is expected to only have one return value but found "
          f"{len(output_node_names)} return values: {output_node_names}.")
    output_node_name = output_node_names[0]
    file_path_nodes = {}
    # When building a tf.function, track files as `saved_model.Asset`s.
    if ops.get_default_graph().building_function:
      asset_tracker = self._maybe_track_assets(graph_def)
      for key in asset_tracker:
        assets_list = [
            array_ops.expand_dims(asset.asset_path, axis=0)
            for asset in asset_tracker[key]
        ]
        file_path_nodes[key] = array_ops.concat(assets_list, axis=0)
    # Add functions used in this Dataset to the function's graph, since they
    # need to follow it around (and for example be added to a SavedModel which
    # references the dataset).
    variant_function = wrap_function.function_from_graph_def(
        graph_def,
        inputs=[],
        outputs=output_node_name + ":0",
        captures=file_path_nodes)
    for used_function in self._functions():
      used_function.function.add_to_graph(variant_function.graph)
    return variant_function
  @abc.abstractmethod
  def _inputs(self):
    """Returns a list of the input datasets of the dataset."""
    raise NotImplementedError(f"{type(self)}._inputs()")
  @property
  def _graph(self):
    return self._graph_attr
  @_graph.setter
  def _graph(self, _):
    raise ValueError("The `_graph` property cannot be modified.")
  # TODO(jsimsa): Change this to be the transitive closure of functions used
  # by this dataset and its inputs.
  def _functions(self):
    """Returns a list of functions associated with this dataset.
    Returns:
      A list of `StructuredFunctionWrapper` objects.
    """
    return []
  def _options(self):
    """Returns the options tensor for this dataset."""
    # pylint: disable=protected-access
    return gen_dataset_ops.get_options(self._variant_tensor)
  @classmethod
  def _options_tensor_to_options(cls, serialized_options):
    """Converts options tensor to tf.data.Options object."""
    options = options_lib.Options()
    if tensor_util.constant_value(serialized_options) is not None:
      pb = dataset_options_pb2.Options.FromString(tensor_util.constant_value(
          serialized_options))
      options._from_proto(pb)  # pylint: disable=protected-access
    return options
  def options(self):
    """Returns the options for this dataset and its inputs.
    Returns:
      A `tf.data.Options` object representing the dataset options.
    """
    if context.executing_eagerly():
      options = self._options_tensor_to_options(self._options())
      options._set_mutable(False)  # pylint: disable=protected-access
      return options
    warnings.warn("To make it possible to preserve tf.data options across "
                  "serialization boundaries, their implementation has moved to "
                  "be part of the TensorFlow graph. As a consequence, the "
                  "options value is in general no longer known at graph "
                  "construction time. Invoking this method in graph mode "
                  "retains the legacy behavior of the original implementation, "
                  "but note that the returned value might not reflect the "
                  "actual value of the options.")
    return self._options_attr
  def _apply_debug_options(self):
    if debug_mode.DEBUG_MODE:
      # Disable autotuning and static optimizations that could introduce
      # parallelism or asynchrony.
      options = options_lib.Options()
      options.autotune.enabled = False
      options.experimental_optimization.filter_parallelization = False
      options.experimental_optimization.map_and_batch_fusion = False
      options.experimental_optimization.map_parallelization = False
      dataset = _OptionsDataset(self, options)
    else:
      dataset = self
    return dataset
  def __iter__(self):
    """Creates an iterator for elements of this dataset.
    The returned iterator implements the Python Iterator protocol.
    Returns:
      An `tf.data.Iterator` for the elements of this dataset.
    Raises:
      RuntimeError: If not inside of tf.function and not executing eagerly.
    """
    if context.executing_eagerly() or ops.inside_function():
      with ops.colocate_with(self._variant_tensor):
        return iterator_ops.OwnedIterator(self)
    else:
      raise RuntimeError("`tf.data.Dataset` only supports Python-style "
                         "iteration in eager mode or within tf.function.")
  def __bool__(self):
    return True  # Required as __len__ is defined
  __nonzero__ = __bool__  # Python 2 backward compatibility
  def __len__(self):
    """Returns the length of the dataset if it is known and finite.
    This method requires that you are running in eager mode, and that the
    length of the dataset is known and non-infinite. When the length may be
    unknown or infinite, or if you are running in graph mode, use
    `tf.data.Dataset.cardinality` instead.
    Returns:
      An integer representing the length of the dataset.
    Raises:
      RuntimeError: If the dataset length is unknown or infinite, or if eager
        execution is not enabled.
    """
    if not context.executing_eagerly():
      raise TypeError("`tf.data.Dataset` only supports `len` in eager mode. "
                      "Use `tf.data.Dataset.cardinality()` instead.")
    length = self.cardinality()
    if length.numpy() == INFINITE:
      raise TypeError("The dataset is infinite.")
    if length.numpy() == UNKNOWN:
      raise TypeError("The dataset length is unknown.")
    return length
  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this dataset.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)
    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).
    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this dataset and specifying the type of individual components.
    """
    raise NotImplementedError(f"{type(self)}.element_spec()")
  def __repr__(self):
    type_ = type(self._dataset if isinstance(self, DatasetV1Adapter) else self)
    return f"<{type_.__name__} element_spec={self.element_spec}>"
  def __debug_string__(self):
    """Returns a string showing the type of the dataset and its inputs.
    This string is intended only for debugging purposes, and may change without
    warning.
    """
    lines = []
    to_process = [(self, 0)]  # Stack of (dataset, depth) pairs.
    while to_process:
      dataset, depth = to_process.pop()
      lines.append("-"*2*depth + repr(dataset))
      to_process.extend([(ds, depth+1) for ds in dataset._inputs()])  # pylint: disable=protected-access
    return "\n".join(lines)
  def as_numpy_iterator(self):
    """Returns an iterator which converts all elements of the dataset to numpy.
    Use `as_numpy_iterator` to inspect the content of your dataset. To see
    element shapes and types, print dataset elements directly instead of using
    `as_numpy_iterator`.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset:
    ...   print(element)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    This method requires that you are running in eager mode and the dataset's
    element_spec contains only `TensorSpec` components.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    1
    2
    3
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> print(list(dataset.as_numpy_iterator()))
    [1, 2, 3]
    `as_numpy_iterator()` will preserve the nested structure of dataset
    elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),
    ...                                               'b': [5, 6]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},
    ...                                       {'a': (2, 4), 'b': 6}]
    True
    Returns:
      An iterable over the elements of the dataset, with their tensors converted
      to numpy arrays.
    Raises:
      TypeError: if an element contains a non-`Tensor` value.
      RuntimeError: if eager execution is not enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`tf.data.Dataset.as_numpy_iterator()` is only "
                         "supported in eager mode.")
    for component_spec in nest.flatten(self.element_spec):
      if not isinstance(
          component_spec,
          (tensor_spec.TensorSpec, ragged_tensor.RaggedTensorSpec,
           sparse_tensor_lib.SparseTensorSpec, structure.NoneTensorSpec)):
        raise TypeError(
            f"`tf.data.Dataset.as_numpy_iterator()` is not supported for "
            f"datasets that produce values of type {component_spec.value_type}")
    return _NumpyIterator(self)
  @property
  def _flat_shapes(self):
    """Returns a list `tf.TensorShapes`s for the element tensor representation.
    Returns:
      A list `tf.TensorShapes`s for the element tensor representation.
    """
    return structure.get_flat_tensor_shapes(self.element_spec)
  @property
  def _flat_types(self):
    """Returns a list `tf.DType`s for the element tensor representation.
    Returns:
      A list `tf.DType`s for the element tensor representation.
    """
    return structure.get_flat_tensor_types(self.element_spec)
  @property
  def _flat_structure(self):
    """Helper for setting `output_shapes` and `output_types` attrs of an op.
    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element. This helper
    function generates these attrs as a keyword argument dictionary, allowing
    `Dataset._variant_tensor` implementations to pass `**self._flat_structure`
    to the op constructor.
    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }
  @property
  def _metadata(self):
    """Helper for generating dataset metadata."""
    metadata = dataset_metadata_pb2.Metadata()
    if self._name:
      metadata.name = _validate_and_encode(self._name)
    return metadata
  @property
  def _common_args(self):
    """Helper for generating arguments that are common across most dataset ops.
    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element, as well as a
    `metadata` argument for additional metadata such as user-defined dataset
    name. This helper function generates common attributes as a keyword argument
    dictionary, allowing `Dataset._variant_tensor` implementations to pass
    `**self._common_args` to the op constructor.
    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "metadata": self._metadata.SerializeToString(),
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }
  @property
  def _type_spec(self):
    return DatasetSpec(self.element_spec)
  @staticmethod
  def from_tensors(tensors, name=None):
    """Creates a `Dataset` with a single element, comprising the given tensors.
    `from_tensors` produces a dataset containing only a single element. To slice
    the input tensor into multiple elements, use `from_tensor_slices` instead.
    >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32)]
    >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
    >>> list(dataset.as_numpy_iterator())
    [(array([1, 2, 3], dtype=int32), b'A')]
    >>> # You can use `from_tensors` to produce a dataset which repeats
    >>> # the same example many times.
    >>> example = tf.constant([1,2,3])
    >>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]
    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).
    Args:
      tensors: A dataset "element". Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).
      name: (Optional.) A name for the tf.data operation.
    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_tensors_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_tensors_op
    return from_tensors_op._from_tensors(tensors, name)
    # pylint: enable=g-import-not-at-top,protected-access
  @staticmethod
  def from_tensor_slices(tensors, name=None):
    """Creates a `Dataset` whose elements are slices of the given tensors.
    The given tensors are sliced along their first dimension. This operation
    preserves the structure of the input tensors, removing the first dimension
    of each tensor and using it as the dataset dimension. All input tensors
    must have the same size in their first dimensions.
    >>> # Slicing a 1D tensor produces scalar tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3]
    >>> # Slicing a 2D tensor produces 1D tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]
    >>> # Slicing a tuple of 1D tensors produces tuple elements containing
    >>> # scalar tensors.
    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
    >>> list(dataset.as_numpy_iterator())
    [(1, 3, 5), (2, 4, 6)]
    >>> # Dictionary structure is also preserved.
    >>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
    ...                                       {'a': 2, 'b': 4}]
    True
    >>> # Two tensors can be combined into one Dataset object.
    >>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
    >>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
    >>> dataset = Dataset.from_tensor_slices((features, labels))
    >>> # Both the features and the labels tensors can be converted
    >>> # to a Dataset object separately and combined after.
    >>> features_dataset = Dataset.from_tensor_slices(features)
    >>> labels_dataset = Dataset.from_tensor_slices(labels)
    >>> dataset = Dataset.zip((features_dataset, labels_dataset))
    >>> # A batched feature and label set can be converted to a Dataset
    >>> # in similar fashion.
    >>> batched_features = tf.constant([[[1, 3], [2, 3]],
    ...                                 [[2, 1], [1, 2]],
    ...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
    >>> batched_labels = tf.constant([['A', 'A'],
    ...                               ['B', 'B'],
    ...                               ['A', 'B']], shape=(3, 2, 1))
    >>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (array([[1, 3],
           [2, 3]], dtype=int32), array([[b'A'],
           [b'A']], dtype=object))
    (array([[2, 1],
           [1, 2]], dtype=int32), array([[b'B'],
           [b'B']], dtype=object))
    (array([[3, 3],
           [3, 2]], dtype=int32), array([[b'A'],
           [b'B']], dtype=object))
    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/data#consuming_numpy_arrays).
    Args:
      tensors: A dataset element, whose components have the same first
        dimension. Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).
      name: (Optional.) A name for the tf.data operation.
    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_tensor_slices_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_tensor_slices_op
    return from_tensor_slices_op._from_tensor_slices(tensors, name)
    # pylint: enable=g-import-not-at-top,protected-access
  class _GeneratorState:
    """Stores outstanding iterators created from a Python generator.
    This class keeps track of potentially multiple iterators that may have
    been created from a generator, e.g. in the case that the dataset is
    repeated, or nested within a parallel computation.
    """
    def __init__(self, generator):
      self._generator = generator
      self._lock = threading.Lock()
      self._next_id = 0  # GUARDED_BY(self._lock)
      self._args = {}
      self._iterators = {}
    def _normalize_id(self, iterator_id):
      # In debug mode, iterator ids may be eagerly-generated np.arrays instead
      # of Tensors. We convert them to scalars to make them hashable.
      if isinstance(iterator_id, np.ndarray):
        return iterator_id.item()
      return iterator_id
    def get_next_id(self, *args):
      with self._lock:
        ret = self._next_id
        self._next_id += 1
      self._args[ret] = args
      # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
      # casting in `py_func()` will create an array of `np.int32` on Windows,
      # leading to a runtime error.
      return np.array(ret, dtype=np.int64)
    def get_iterator(self, iterator_id):
      iterator_id = self._normalize_id(iterator_id)
      try:
        return self._iterators[iterator_id]
      except KeyError:
        iterator = iter(self._generator(*self._args.pop(iterator_id)))
        self._iterators[iterator_id] = iterator
        return iterator
    def iterator_completed(self, iterator_id):
      del self._iterators[self._normalize_id(iterator_id)]
  @staticmethod
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None,
                     name=None):
    """Creates a `Dataset` whose elements are generated by `generator`.
    Note: The current implementation of `Dataset.from_generator()` uses
    `tf.numpy_function` and inherits the same constraints. In particular, it
    requires the dataset and iterator related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. In particular, using `from_generator` will
    preclude the use of tf.data service for scaling out dataset processing.
    The body of `generator` will not be serialized in a `GraphDef`, and you
    should not use this method if you need to serialize your model and restore
    it in a different environment.
    The `generator` argument must be a callable object that returns
    an object that supports the `iter()` protocol (e.g. a generator function).
    The elements generated by `generator` must be compatible with either the
    given `output_signature` argument or with the given `output_types` and
    (optionally) `output_shapes` arguments, whichever was specified.
    The recommended way to call `from_generator` is to use the
    `output_signature` argument. In this case the output will be assumed to
    consist of objects with the classes, shapes and types defined by
    `tf.TypeSpec` objects from `output_signature` argument:
    >>> def gen():
    ...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    ...   yield 42, ragged_tensor
    >>>
    >>> dataset = tf.data.Dataset.from_generator(
    ...      gen,
    ...      output_signature=(
    ...          tf.TensorSpec(shape=(), dtype=tf.int32),
    ...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
    >>>
    >>> list(dataset.take(1))
    [(<tf.Tensor: shape=(), dtype=int32, numpy=42>,
    <tf.RaggedTensor [[1, 2], [3]]>)]
    There is also a deprecated way to call `from_generator` by either with
    `output_types` argument alone or together with `output_shapes` argument.
    In this case the output of the function will be assumed to consist of
    `tf.Tensor` objects with the types defined by `output_types` and with the
    shapes which are either unknown or defined by `output_shapes`.
    Note: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.
    Note: While the `output_signature` parameter makes it possible to yield
    `Dataset` elements, the scope of `Dataset.from_generator()` should be
    limited to logic that cannot be expressed through tf.data operations. Using
    tf.data operations within the generator function is an anti-pattern and may
    result in incremental memory growth.
    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take no
        arguments; otherwise it must take as many arguments as there are values
        in `args`.
      output_types: (Optional.) A (nested) structure of `tf.DType` objects
        corresponding to each component of an element yielded by `generator`.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element yielded by
        `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.
      output_signature: (Optional.) A (nested) structure of `tf.TypeSpec`
        objects corresponding to each component of an element yielded by
        `generator`.
      name: (Optional.) A name for the tf.data operations used by
        `from_generator`.
    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_generator_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_generator_op
    return from_generator_op._from_generator(generator, output_types,
                                             output_shapes, args,
                                             output_signature, name)
    # pylint: enable=g-import-not-at-top,protected-access
  @staticmethod
  def range(*args, **kwargs):
    """Creates a `Dataset` of a step-separated range of values.
    >>> list(Dataset.range(5).as_numpy_iterator())
    [0, 1, 2, 3, 4]
    >>> list(Dataset.range(2, 5).as_numpy_iterator())
    [2, 3, 4]
    >>> list(Dataset.range(1, 5, 2).as_numpy_iterator())
    [1, 3]
    >>> list(Dataset.range(1, 5, -2).as_numpy_iterator())
    []
    >>> list(Dataset.range(5, 1).as_numpy_iterator())
    []
    >>> list(Dataset.range(5, 1, -2).as_numpy_iterator())
    [5, 3]
    >>> list(Dataset.range(2, 5, output_type=tf.int32).as_numpy_iterator())
    [2, 3, 4]
    >>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())
    [1.0, 3.0]
    Args:
      *args: follows the same semantics as python's range.
        len(args) == 1 -> start = 0, stop = args[0], step = 1.
        len(args) == 2 -> start = args[0], stop = args[1], step = 1.
        len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
      **kwargs:
        - output_type: Its expected dtype. (Optional, default: `tf.int64`).
        - name: (Optional.) A name for the tf.data operation.
    Returns:
      Dataset: A `RangeDataset`.
    Raises:
      ValueError: if len(args) == 0.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> range_op ->
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import range_op
    return range_op._range(*args, **kwargs)
    # pylint: enable=g-import-not-at-top,protected-access
  @staticmethod
  def zip(*args, datasets=None, name=None):
    """Creates a `Dataset` by zipping together the given datasets.
    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be a (nested) structure of `Dataset` objects. The supported
    nesting mechanisms are documented
    [here] (https://www.tensorflow.org/guide/data#dataset_structure).
    >>> # The datasets or nested structure of datasets `*args` argument
    >>> # determines the structure of elements in the resulting dataset.
    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
    >>> ds = tf.data.Dataset.zip(a, b)
    >>> list(ds.as_numpy_iterator())
    [(1, 4), (2, 5), (3, 6)]
    >>> ds = tf.data.Dataset.zip(b, a)
    >>> list(ds.as_numpy_iterator())
    [(4, 1), (5, 2), (6, 3)]
    >>>
    >>> # The `datasets` argument may contain an arbitrary number of datasets.
    >>> c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],
    ...                                            #       [9, 10],
    ...                                            #       [11, 12] ]
    >>> ds = tf.data.Dataset.zip(a, b, c)
    >>> for element in ds.as_numpy_iterator():
    ...   print(element)
    (1, 4, array([7, 8]))
    (2, 5, array([ 9, 10]))
    (3, 6, array([11, 12]))
    >>>
    >>> # The number of elements in the resulting dataset is the same as
    >>> # the size of the smallest dataset in `datasets`.
    >>> d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]
    >>> ds = tf.data.Dataset.zip(a, d)
    >>> list(ds.as_numpy_iterator())
    [(1, 13), (2, 14)]
    Args:
      *args: Datasets or nested structures of datasets to zip together. This
        can't be set if `datasets` is set.
      datasets: A (nested) structure of datasets. This can't be set if `*args`
        is set. Note that this exists only for backwards compatibility and it is
        preferred to use *args.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> zip_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import zip_op
    if not args and datasets is None:
      raise TypeError("Must pass at least one dataset to `zip`.")
    if args and datasets is not None:
      raise TypeError("Both `*args` and `datasets` cannot be set.")
    if len(args) == 1:
      datasets = args[0]
    elif len(args) > 1:
      datasets = args
    return zip_op._zip(datasets, name)
    # pylint: enable=g-import-not-at-top,protected-access
  def concatenate(self, dataset, name=None):
    """Creates a `Dataset` by concatenating the given dataset with this dataset.
    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
    >>> ds = a.concatenate(b)
    >>> list(ds.as_numpy_iterator())
    [1, 2, 3, 4, 5, 6, 7]
    >>> # The input dataset and dataset to be concatenated should have
    >>> # compatible element specs.
    >>> c = tf.data.Dataset.zip((a, b))
    >>> a.concatenate(c)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and (tf.int64, tf.int64)
    >>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    >>> a.concatenate(d)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and <dtype: 'string'>
    Args:
      dataset: `Dataset` to be concatenated.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # concatenate_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import concatenate_op
    return concatenate_op._concatenate(self, dataset, name)
    # pylint: enable=g-import-not-at-top,protected-access
  @staticmethod
  def counter(start=0, step=1, dtype=dtypes.int64, name=None):
    """Creates a `Dataset` that counts from `start` in steps of size `step`.
    Unlike `tf.data.Dataset.range`, which stops at some ending number,
    `tf.data.Dataset.counter` produces elements indefinitely.
    >>> dataset = tf.data.experimental.Counter().take(5)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2, 3, 4]
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int64, name=None)
    >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)
    >>> dataset = tf.data.experimental.Counter(start=2).take(5)
    >>> list(dataset.as_numpy_iterator())
    [2, 3, 4, 5, 6]
    >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
    >>> list(dataset.as_numpy_iterator())
    [2, 7, 12, 17, 22]
    >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
    >>> list(dataset.as_numpy_iterator())
    [10, 9, 8, 7, 6]
    Args:
      start: (Optional.) The starting value for the counter. Defaults to 0.
      step: (Optional.) The step size for the counter. Defaults to 1.
      dtype: (Optional.) The data type for counter elements. Defaults to
        `tf.int64`.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A `Dataset` of scalar `dtype` elements.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> counter_op
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import counter_op
    return counter_op._counter(start, step, dtype, name=name)
    # pylint: enable=g-import-not-at-top,protected-access
  def rebatch(self, batch_size, drop_remainder=False, name=None):
    """Creates a `Dataset` that rebatches the elements from this dataset.
    `rebatch(N)` is functionally equivalent to `unbatch().batch(N)`, but is
    more efficient, performing one copy instead of two.
    >>> ds = tf.data.Dataset.range(6)
    >>> ds = ds.batch(2)
    >>> ds = ds.rebatch(3)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]
    >>> ds = tf.data.Dataset.range(7)
    >>> ds = ds.batch(4)
    >>> ds = ds.rebatch(3)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5]), array([6])]
    >>> ds = tf.data.Dataset.range(7)
    >>> ds = ds.batch(2)
    >>> ds = ds.rebatch(3, drop_remainder=True)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]
    If the `batch_size` argument is a list, `rebatch` cycles through the list
    to determine the size of each batch.
    >>> ds = tf.data.Dataset.range(8)
    >>> ds = ds.batch(4)
    >>> ds = ds.rebatch([2, 1, 1])
    >>> list(ds.as_numpy_iterator())
    [array([0, 1]), array([2]), array([3]), array([4, 5]), array([6]),
    array([7])]
    Args:
      batch_size: A `tf.int64` scalar or vector, representing the size of
        batches to produce. If this argument is a vector, these values are
        cycled through in round robin fashion.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size[cycle_index]` elements; the default behavior is not to drop
        the smaller batch.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A `Dataset` of scalar `dtype` elements.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> rebatch_op ->
    # rebatch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import rebatch_op
    return rebatch_op._rebatch(self, batch_size, drop_remainder, name=name)
    # pylint: enable=g-import-not-at-top,protected-access
  def prefetch(self, buffer_size, name=None):
    """Creates a `Dataset` that prefetches elements from this dataset.
    Most dataset input pipelines should end with a call to `prefetch`. This
    allows later elements to be prepared while the current element is being
    processed. This often improves latency and throughput, at the cost of
    using additional memory to store prefetched elements.
    Note: Like other `Dataset` methods, prefetch operates on the
    elements of the input dataset. It has no concept of examples vs. batches.
    `examples.prefetch(2)` will prefetch two elements (2 examples),
    while `examples.batch(20).prefetch(2)` will prefetch 2 elements
    (2 batches, of 20 examples each).
    >>> dataset = tf.data.Dataset.range(3)
    >>> dataset = dataset.prefetch(2)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2]
    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
        number of elements that will be buffered when prefetching. If the value
        `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
      name: Optional. A name for the tf.data transformation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    return prefetch_op._prefetch(  # pylint: disable=protected-access
        self, buffer_size, name=name)
  @staticmethod
  def list_files(file_pattern, shuffle=None, seed=None, name=None):
    """A dataset of all files matching one or more glob patterns.
    The `file_pattern` argument should be a small number of glob patterns.
    If your filenames have already been globbed, use
    `Dataset.from_tensor_slices(filenames)` instead, as re-globbing every
    filename with `list_files` may result in poor performance with remote
    storage systems.
    Note: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.
    Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset
      would produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py
    Args:
      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      name: Optional. A name for the tf.data operations used by `list_files`.
    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    """
    with ops.name_scope("list_files"):
      if shuffle is None:
        shuffle = True
      file_pattern = ops.convert_to_tensor(
          file_pattern, dtype=dtypes.string, name="file_pattern")
      matching_files = gen_io_ops.matching_files(file_pattern)
      # Raise an exception if `file_pattern` does not match any files.
      condition = math_ops.greater(array_ops.shape(matching_files)[0], 0,
                                   name="match_not_empty")
      message = math_ops.add(
          "No files matched pattern: ",
          string_ops.reduce_join(file_pattern, separator=", "), name="message")
      assert_not_empty = control_flow_assert.Assert(
          condition, [message], summarize=1, name="assert_not_empty")
      with ops.control_dependencies([assert_not_empty]):
        matching_files = array_ops.identity(matching_files)
      # TODO(b/240947712): Remove lazy import after this method is factored out.
      # Loaded lazily due to a circular dependency (dataset_ops ->
      # from_tensor_slices_op -> dataset_ops).
      # pylint: disable=g-import-not-at-top,protected-access
      from tensorflow.python.data.ops import from_tensor_slices_op
      dataset = from_tensor_slices_op._TensorSliceDataset(
          matching_files, is_files=True, name=name)
      # pylint: enable=g-import-not-at-top,protected-access
      if issubclass(Dataset, DatasetV1):
        dataset = DatasetV1Adapter(dataset)
      if shuffle:
        # NOTE(mrry): The shuffle buffer size must be greater than zero, but the
        # list of files might be empty.
        buffer_size = math_ops.maximum(
            array_ops.shape(matching_files, out_type=dtypes.int64)[0], 1)
        dataset = dataset.shuffle(buffer_size, seed=seed, name=name)
      return dataset
  def repeat(self, count=None, name=None):
    """Repeats this dataset so each original value is seen `count` times.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.repeat(3)
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3, 1, 2, 3, 1, 2, 3]
    Note: If the input dataset depends on global state (e.g. a random number
    generator) or its output is non-deterministic (e.g. because of upstream
    `shuffle`), then different repetitions may produce different elements.
    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior (if
        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> repeat_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
    from tensorflow.python.data.ops import repeat_op
    return repeat_op._repeat(self, count, name)
    # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name
  def enumerate(self, start=0, name=None):
    """Enumerates the elements of this dataset.
    It is similar to python's `enumerate`.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.enumerate(start=5)
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (5, 1)
    (6, 2)
    (7, 3)
    >>> # The (nested) structure of the input dataset determines the
    >>> # structure of elements in the resulting dataset.
    >>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
    >>> dataset = dataset.enumerate()
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (0, array([7, 8], dtype=int32))
    (1, array([ 9, 10], dtype=int32))
    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
        enumeration.
      name: Optional. A name for the tf.data operations used by `enumerate`.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    range_dataset = Dataset.range(start, max_value, name=name)
    # Replicate the range component so that each split is enumerated
    # independently. This avoids the need for prohibitively expensive
    # cross-split coordination.
    range_dataset = _apply_rewrite(range_dataset, "replicate_on_split")
    return Dataset.zip((range_dataset, self), name=name)
  def shuffle(self,
              buffer_size,
              seed=None,
              reshuffle_each_iteration=None,
              name=None):
    """Randomly shuffles the elements of this dataset.
    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.
    For instance, if your dataset contains 10,000 elements but `buffer_size` is
    set to 1,000, then `shuffle` will initially select a random element from
    only the first 1,000 elements in the buffer. Once an element is selected,
    its space in the buffer is replaced by the next (i.e. 1,001-st) element,
    maintaining the 1,000 element buffer.
    `reshuffle_each_iteration` controls whether the shuffle order should be
    different for each epoch. In TF 1.X, the idiomatic way to create epochs
    was through the `repeat` transformation:
    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 2, 0]
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 0, 2]
    ```
    In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
    possible to also create epochs through Python iteration:
    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 2, 0]
    ```
    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    ```
    ### Fully shuffling all the data
    To shuffle an entire dataset, set `buffer_size=dataset.cardinality(). This
    is equivalent to setting the `buffer_size` equal to the number of elements
    in the dataset, resulting in uniform shuffle.
    Note: `shuffle(dataset.cardinality())` loads the full dataset into memory so
    that it can be shuffled. This will cause a memory overflow (OOM) error if
    the dataset is too large, so full-shuffle should only be used for datasets
    that are known to fit in the memory, such as datasets of filenames or other
    small datasets.
    ```python
    dataset = tf.data.Dataset.range(20)
    dataset = dataset.shuffle(dataset.cardinality())
    # [18, 4, 9, 2, 17, 8, 5, 10, 0, 6, 16, 3, 19, 7, 14, 11, 15, 13, 12, 1]
    ```
    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample. To
        uniformly shuffle the entire dataset, use
        `buffer_size=dataset.cardinality()`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    return shuffle_op._shuffle(  # pylint: disable=protected-access
        self, buffer_size, seed, reshuffle_each_iteration, name=name)
  def cache(self, filename="", name=None):
    """Caches the elements in this dataset.
    The first time the dataset is iterated over, its elements will be cached
    either in the specified file or in memory. Subsequent iterations will
    use the cached data.
    Note: To guarantee that the cache gets finalized, the input dataset must be
    iterated through in its entirety, until it raises StopIteration. Otherwise,
    subsequent iterations may not use cached data.
    >>> dataset = tf.data.Dataset.range(5)
    >>> dataset = dataset.map(lambda x: x**2)
    >>> dataset = dataset.cache()
    >>> # The first time reading through the data will generate the data using
    >>> # `range` and `map`.
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 4, 9, 16]
    >>> # Subsequent iterations read from the cache.
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 4, 9, 16]
    When caching to a file, the cached data will persist across runs. Even the
    first iteration through the data will read from the cache file. Changing
    the input pipeline before the call to `.cache()` will have no effect until
    the cache file is removed or the filename is changed.
    ```python
    dataset = tf.data.Dataset.range(5)
    dataset = dataset.cache("/path/to/file")
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.cache("/path/to/file")  # Same file!
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    ```
    Note: `cache` will produce exactly the same elements during each iteration
    through the dataset. If you wish to randomize the iteration order, make sure
    to call `shuffle` *after* calling `cache`.
    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching elements in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> cache_op ->
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import cache_op
    return cache_op._cache(self, filename, name)
    # pylint: enable=g-import-not-at-top,protected-access
  def take(self, count, name=None):
    """Creates a `Dataset` with at most `count` elements from this dataset.
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.take(3)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2]
    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be taken to form the new dataset.
        If `count` is -1, or if `count` is greater than the size of this
        dataset, the new dataset will contain all elements of this dataset.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # take_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import take_op
    return take_op._take(self, count, name=name)
    # pylint: enable=g-import-not-at-top,protected-access
  def skip(self, count, name=None):
    """Creates a `Dataset` that skips `count` elements from this dataset.
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.skip(7)
    >>> list(dataset.as_numpy_iterator())
    [7, 8, 9]
    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be skipped to form the new dataset.
        If `count` is greater than the size of this dataset, the new dataset
        will contain no elements.  If `count` is -1, skips the entire dataset.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # skip_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import skip_op
    return skip_op._skip(self, count, name)
    # pylint: enable=g-import-not-at-top,protected-access
  def shard(self, num_shards, index, name=None):
    """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
    `shard` is deterministic. The Dataset produced by `A.shard(n, i)` will
    contain all elements of A whose index mod n = i.
    >>> A = tf.data.Dataset.range(10)
    >>> B = A.shard(num_shards=3, index=0)
    >>> list(B.as_numpy_iterator())
    [0, 3, 6, 9]
    >>> C = A.shard(num_shards=3, index=1)
    >>> list(C.as_numpy_iterator())
    [1, 4, 7]
    >>> D = A.shard(num_shards=3, index=2)
    >>> list(D.as_numpy_iterator())
    [2, 5, 8]
    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.
    When reading a single input file, you can shard elements as follows:
    ```python
    d = tf.data.TFRecordDataset(input_file)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```
    Important caveats:
    - Be sure to shard before you use any randomizing operator (such as
      shuffle).
    - Generally it is best if the shard operator is used early in the dataset
      pipeline. For example, when reading from a set of TFRecord files, shard
      before converting the dataset to input samples. This avoids reading every
      file on every worker. The following is an example of an efficient
      sharding strategy within a complete pipeline:
    ```python
    d = Dataset.list_files(pattern, shuffle=False)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```
    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    Raises:
      InvalidArgumentError: if `num_shards` or `index` are illegal values.
        Note: error checking is done on a best-effort basis, and errors aren't
        guaranteed to be caught upon dataset creation. (e.g. providing in a
        placeholder tensor bypasses the early checking, and will instead result
        in an error during a session.run call.)
    """
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import shard_op
    return shard_op._shard(self, num_shards, index, name=name)
    # pylint: enable=g-import-not-at-top,protected-access
  def save(self,
           path,
           compression=None,
           shard_func=None,
           checkpoint_args=None):
    """Saves the content of the given dataset.
      Example usage:
      >>> import tempfile
      >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
      >>> # Save a dataset
      >>> dataset = tf.data.Dataset.range(2)
      >>> dataset.save(path)
      >>> new_dataset = tf.data.Dataset.load(path)
      >>> for elem in new_dataset:
      ...   print(elem)
      tf.Tensor(0, shape=(), dtype=int64)
      tf.Tensor(1, shape=(), dtype=int64)
      The saved dataset is saved in multiple file "shards". By default, the
      dataset output is divided to shards in a round-robin fashion but custom
      sharding can be specified via the `shard_func` function. For example, you
      can save the dataset to using a single shard as follows:
      ```python
      dataset = make_dataset()
      def custom_shard_func(element):
        return np.int64(0)
      dataset.save(
          path="/path/to/data", ..., shard_func=custom_shard_func)
      ```
      To enable checkpointing, pass in `checkpoint_args` to the `save` method
      as follows:
      ```python
      dataset = tf.data.Dataset.range(100)
      save_dir = "..."
      checkpoint_prefix = "..."
      step_counter = tf.Variable(0, trainable=False)
      checkpoint_args = {
        "checkpoint_interval": 50,
        "step_counter": step_counter,
        "directory": checkpoint_prefix,
        "max_to_keep": 20,
      }
      dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)
      ```
      NOTE: The directory layout and file format used for saving the dataset is
      considered an implementation detail and may change. For this reason,
      datasets saved through `tf.data.Dataset.save` should only be consumed
      through `tf.data.Dataset.load`, which is guaranteed to be
      backwards compatible.
    Args:
     path: Required. A directory to use for saving the dataset.
     compression: Optional. The algorithm to use to compress data when writing
          it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
     shard_func: Optional. A function to control the mapping of dataset
          elements to file shards. The function is expected to map elements of
          the input dataset to int64 shard IDs. If present, the function will be
          traced and executed as graph computation.
     checkpoint_args: Optional args for checkpointing which will be passed into
          the `tf.train.CheckpointManager`. If `checkpoint_args` are not
          specified, then checkpointing will not be performed. The `save()`
          implementation creates a `tf.train.Checkpoint` object internally, so
          users should not set the `checkpoint` argument in `checkpoint_args`.
    Returns:
      An operation which when executed performs the save. When writing
      checkpoints, returns None. The return value is useful in unit tests.
    Raises:
      ValueError if `checkpoint` is passed into `checkpoint_args`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> save_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import save_op
    return save_op._save(self, path, compression, shard_func, checkpoint_args)
    # pylint: enable=g-import-not-at-top,protected-access
  @staticmethod
  def load(path, element_spec=None, compression=None, reader_func=None):
    """Loads a previously saved dataset.
    Example usage:
    >>> import tempfile
    >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
    >>> # Save a dataset
    >>> dataset = tf.data.Dataset.range(2)
    >>> tf.data.Dataset.save(dataset, path)
    >>> new_dataset = tf.data.Dataset.load(path)
    >>> for elem in new_dataset:
    ...   print(elem)
    tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor(1, shape=(), dtype=int64)
    If the default option of sharding the saved dataset was used, the element
    order of the saved dataset will be preserved when loading it.
    The `reader_func` argument can be used to specify a custom order in which
    elements should be loaded from the individual shards. The `reader_func` is
    expected to take a single argument -- a dataset of datasets, each containing
    elements of one of the shards -- and return a dataset of elements. For
    example, the order of shards can be shuffled when loading them as follows:
    ```python
    def custom_reader_func(datasets):
      datasets = datasets.shuffle(NUM_SHARDS)
      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)
    dataset = tf.data.Dataset.load(
        path="/path/to/data", ..., reader_func=custom_reader_func)
    ```
    Args:
      path: Required. A path pointing to a previously saved dataset.
      element_spec: Optional. A nested structure of `tf.TypeSpec` objects
        matching the structure of an element of the saved dataset and specifying
        the type of individual element components. If not provided, the nested
        structure of `tf.TypeSpec` saved with the saved dataset is used. Note
        that this argument is required in graph mode.
      compression: Optional. The algorithm to use to decompress the data when
        reading it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
      reader_func: Optional. A function to control how to read data from shards.
        If present, the function will be traced and executed as graph
        computation.
    Returns:
      A `tf.data.Dataset` instance.
    Raises:
      FileNotFoundError: If `element_spec` is not specified and the saved nested
        structure of `tf.TypeSpec` can not be located with the saved dataset.
      ValueError: If `element_spec` is not specified and the method is executed
        in graph mode.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> load_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import load_op
    return load_op._load(
        path=path,
        element_spec=element_spec,
        compression=compression,
        reader_func=reader_func)
    # pylint: enable=g-import-not-at-top,protected-access
  def batch(self,
            batch_size,
            drop_remainder=False,
            num_parallel_calls=None,
            deterministic=None,
            name=None):
    """Combines consecutive elements of this dataset into batches.
    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]
    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3, drop_remainder=True)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]
    The components of the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.
    Note: If your program requires data to have a statically known shape (e.g.,
    when using XLA), you should use `drop_remainder=True`. Without
    `drop_remainder=True` the shape of the output dataset will have an unknown
    leading dimension due to the possibility of a smaller final batch.
    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number of batches to compute asynchronously in
        parallel.
        If not specified, batches will be computed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available resources.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.deterministic` option (`True` by default) controls the
        behavior.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> batch_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
    from tensorflow.python.data.ops import batch_op
    return batch_op._batch(self, batch_size, drop_remainder, num_parallel_calls,
                           deterministic, name)
    # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name
  def padded_batch(self,
                   batch_size,
                   padded_shapes=None,
                   padding_values=None,
                   drop_remainder=False,
                   name=None):
    """Combines consecutive elements of this dataset into padded batches.
    This transformation combines multiple consecutive elements of the input
    dataset into a single element.
    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.
    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padded_shapes`. The `padded_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:
    * If the dimension is a constant, the component will be padded out to that
      length in that dimension.
    * If the dimension is unknown, the component will be padded out to the
      maximum length of all elements in that dimension.
    >>> A = (tf.data.Dataset
    ...      .range(1, 5, output_type=tf.int32)
    ...      .map(lambda x: tf.fill([x], x)))
    >>> # Pad to the smallest per-batch size that fits all elements.
    >>> B = A.padded_batch(2)
    >>> for element in B.as_numpy_iterator():
    ...   print(element)
    [[1 0]
     [2 2]]
    [[3 3 3 0]
     [4 4 4 4]]
    >>> # Pad to a fixed size.
    >>> C = A.padded_batch(2, padded_shapes=5)
    >>> for element in C.as_numpy_iterator():
    ...   print(element)
    [[1 0 0 0 0]
     [2 2 0 0 0]]
    [[3 3 3 0 0]
     [4 4 4 4 0]]
    >>> # Pad with a custom value.
    >>> D = A.padded_batch(2, padded_shapes=5, padding_values=-1)
    >>> for element in D.as_numpy_iterator():
    ...   print(element)
    [[ 1 -1 -1 -1 -1]
     [ 2  2 -1 -1 -1]]
    [[ 3  3  3 -1 -1]
     [ 4  4  4  4 -1]]
    >>> # Components of nested elements can be padded independently.
    >>> elements = [([1, 2, 3], [10]),
    ...             ([4, 5], [11, 12])]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: iter(elements), (tf.int32, tf.int32))
    >>> # Pad the first component of the tuple to length 4, and the second
    >>> # component to the smallest size that fits.
    >>> dataset = dataset.padded_batch(2,
    ...     padded_shapes=([4], [None]),
    ...     padding_values=(-1, 100))
    >>> list(dataset.as_numpy_iterator())
    [(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),
      array([[ 10, 100], [ 11,  12]], dtype=int32))]
    >>> # Pad with a single value and multiple components.
    >>> E = tf.data.Dataset.zip((A, A)).padded_batch(2, padding_values=-1)
    >>> for element in E.as_numpy_iterator():
    ...   print(element)
    (array([[ 1, -1],
           [ 2,  2]], dtype=int32), array([[ 1, -1],
           [ 2,  2]], dtype=int32))
    (array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32), array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32))
    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.sparse.SparseTensor`.
    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or
        `tf.int64` vector tensor-like objects representing the shape to which
        the respective component of each input element should be padded prior
        to batching. Any unknown dimensions will be padded to the maximum size
        of that dimension in each batch. If unset, all dimensions of all
        components are padded to the maximum size in the batch. `padded_shapes`
        must be set if any component has an unknown rank.
      padding_values: (Optional.) A (nested) structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the respective
        components. None represents that the (nested) structure should be padded
        with default values.  Defaults are `0` for numeric types and the empty
        string for string types. The `padding_values` should have the same
        (nested) structure as the input dataset. If `padding_values` is a single
        element and the input dataset has multiple components, then the same
        `padding_values` will be used to pad every component of the dataset.
        If `padding_values` is a scalar, then its value will be broadcasted
        to match the shape of each component.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      name: (Optional.) A name for the tf.data operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    Raises:
      ValueError: If a component has an unknown rank, and the `padded_shapes`
        argument is not set.
      TypeError: If a component is of an unsupported type. The list of supported
        types is documented in
        https://www.tensorflow.org/guide/data#dataset_structure.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # padded_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import padded_batch_op
    return padded_batch_op._padded_batch(self, batch_size, padded_shapes,
                                         padding_values, drop_remainder, name)
    # pylint: enable=g-import-not-at-top,protected-access
  def ragged_batch(self,
                   batch_size,
                   drop_remainder=False,
                   row_splits_dtype=dtypes.int64,
                   name=None):
    """Combines consecutive elements of this dataset into `tf.RaggedTensor`s.
    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.
    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes:
    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape` is
    fully defined, then it is batched as normal.
    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape`
    contains one or more axes with unknown size (i.e., `shape[i]=None`), then
    the output will contain a `tf.RaggedTensor` that is ragged up to any of such
    dimensions.
    *  If an input element is a `tf.RaggedTensor` or any other type, then it is
    batched as normal.
    Example:
    >>> dataset = tf.data.Dataset.range(6)
    >>> dataset = dataset.map(lambda x: tf.range(x))
    >>> dataset.element_spec.shape
    TensorShape([None])
    >>> dataset = dataset.ragged_batch(2)
    >>> for batch in dataset:
    ...   print(batch)
    <tf.RaggedTensor [[], [0]]>
    <tf.RaggedTensor [[0, 1], [0, 1, 2]]>
    <tf.RaggedTensor [[0, 1, 2, 3], [0, 1, 2, 3, 4]]>
    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      row_splits_dtype: The dtype that should be used for the `row_splits` of
        any new ragged tensors.  Existing `tf.RaggedTensor` elements do not have
        their row_splits dtype changed.
      name: (Optional.) A string indicating a name for the `tf.data` operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # ragged_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import ragged_batch_op
    return ragged_batch_op._ragged_batch(self, batch_size, drop_remainder,
                                         row_splits_dtype, name)
    # pylint: enable=g-import-not-at-top,protected-access
  def sparse_batch(self, batch_size, row_shape, name=None):
    """Combines consecutive elements into `tf.sparse.SparseTensor`s.
    Like `Dataset.padded_batch()`, this transformation combines multiple
    consecutive elements of the dataset, which might have different
    shapes, into a single element. The resulting element has three
    components (`indices`, `values`, and `dense_shape`), which
    comprise a `tf.sparse.SparseTensor` that represents the same data. The
    `row_shape` represents the dense shape of each row in the
    resulting `tf.sparse.SparseTensor`, to which the effective batch size is
    prepended. For example:
    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }
    a.apply(tf.data.experimental.dense_to_sparse_batch(
        batch_size=2, row_shape=[6])) ==
    {
        ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
         ['a', 'b', 'c', 'a', 'b'],                 # values
         [2, 6]),                                   # dense_shape
        ([[0, 0], [0, 1], [0, 2], [0, 3]],
         ['a', 'b', 'c', 'd'],
         [1, 6])
    }
    ```
    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object
        representing the equivalent dense shape of a row in the resulting
        `tf.sparse.SparseTensor`. Each element of this dataset must have the
        same rank as `row_shape`, and must have size less than or equal to
        `row_shape` in each dimension.
      name: (Optional.) A string indicating a name for the `tf.data` operation.
    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # sparse_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import sparse_batch_op
    return sparse_batch_op._sparse_batch(self, batch_size, row_shape, name)
    # pylint: disable=g-import-not-at-top,protected-access
  def map(self,
          map_func,
          num_parallel_calls=None,
          deterministic=None,
          name=None):
    """Maps `map_func` across the elements of this dataset.
    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input. `map_func` can be used to change both
    the values and the structure of a dataset's elements. Supported structure
    constructs are documented
    [here](https://www.tensorflow.org/guide/data#dataset_structure).
    For example, `map` can be used for adding 1 to each element, or projecting a
    subset of element components.
    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> dataset = dataset.map(lambda x: x + 1)
    >>> list(dataset.as_numpy_iterator())
    [2, 3, 4, 5, 6]
    The input signature of `map_func` is determined by the structure of each
    element in this dataset.
    >>> dataset = Dataset.range(5)
    >>> # `map_func` takes a single argument of type `tf.Tensor` with the same
    >>> # shape and dtype.
    >>> result = dataset.map(lambda x: x + 1)
    >>> # Each element is a tuple containing two `tf.Tensor` objects.
    >>> elements = [(1, "foo"), (2, "bar"), (3, "baz")]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, (tf.int32, tf.string))
    >>> # `map_func` takes two arguments of type `tf.Tensor`. This function
    >>> # projects out just the first component.
    >>> result = dataset.map(lambda x_int, y_str: x_int)
    >>> list(result.as_numpy_iterator())
    [1, 2, 3]
    >>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    >>> elements =  ([{"a": 1, "b": "foo"},
    ...               {"a": 2, "b": "bar"},
    ...               {"a": 3, "b": "baz"}])
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, {"a": tf.int32, "b": tf.string})
    >>> # `map_func` takes a single argument of type `dict` with the same keys
    >>> # as the elements.
    >>> result = dataset.map(lambda d: str(d["a"]) + d["b"])
    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.
    >>> dataset = tf.data.Dataset.range(3)
    >>> # `map_func` returns two `tf.Tensor` objects.
    >>> def g(x):
    ...   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
    >>> result = dataset.map(g)
    >>> result.element_spec
    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(3,), \
