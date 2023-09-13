@tf_export("Graph")
class Graph(pywrap_tf_session.PyGraph):
  """A TensorFlow computation, represented as a dataflow graph.
  Graphs are used by `tf.function`s to represent the function's computations.
  Each graph contains a set of `tf.Operation` objects, which represent units of
  computation; and `tf.Tensor` objects, which represent the units of data that
  flow between operations.
  ### Using graphs directly (deprecated)
  A `tf.Graph` can be constructed and used directly without a `tf.function`, as
  was required in TensorFlow 1, but this is deprecated and it is recommended to
  use a `tf.function` instead. If a graph is directly used, other deprecated
  TensorFlow 1 classes are also required to execute the graph, such as a
  `tf.compat.v1.Session`.
  A default graph can be registered with the `tf.Graph.as_default` context
  manager. Then, operations will be added to the graph instead of being executed
  eagerly. For example:
  ```python
  g = tf.Graph()
  with g.as_default():
    # Define operations and tensors in `g`.
    c = tf.constant(30.0)
    assert c.graph is g
  ```
  `tf.compat.v1.get_default_graph()` can be used to obtain the default graph.
  Important note: This class *is not* thread-safe for graph construction. All
  operations should be created from a single thread, or external
  synchronization must be provided. Unless otherwise specified, all methods
  are not thread-safe.
  A `Graph` instance supports an arbitrary number of "collections"
  that are identified by name. For convenience when building a large
  graph, collections can store groups of related objects: for
  example, the `tf.Variable` uses a collection (named
  `tf.GraphKeys.GLOBAL_VARIABLES`) for
  all variables that are created during the construction of a graph. The caller
  may define additional collections by specifying a new name.
  """
  def __init__(self):
    """Creates a new, empty Graph."""
    super().__init__()
    # Protects core state that can be returned via public accessors.
    # Thread-safety is provided on a best-effort basis to support buggy
    # programs, and is not guaranteed by the public `tf.Graph` API.
    #
    # NOTE(mrry): This does not protect the various stacks. A warning will
    # be reported if these are used from multiple threads
    self._lock = threading.RLock()
    # The group lock synchronizes Session.run calls with methods that create
    # and mutate ops (e.g. Graph.create_op()). This synchronization is
    # necessary because it's illegal to modify an operation after it's been run.
    # The group lock allows any number of threads to mutate ops at the same time
    # but if any modification is going on, all Session.run calls have to wait.
    # Similarly, if one or more Session.run calls are going on, all mutate ops
    # have to wait until all Session.run calls have finished.
    self._group_lock = lock_util.GroupLock(num_groups=2)
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    self._stack_state_is_thread_local = False
    self._thread_local = threading.local()
    # Functions that will be applied to choose a device if none is specified.
    # In TF2.x or after switch_to_thread_local(),
    # self._thread_local._device_function_stack is used instead.
    self._graph_device_function_stack = traceable_stack.TraceableStack()
    # Default original_op applied to new ops.
    self._default_original_op = None
    # Current control flow context. It could be either CondContext or
    # WhileContext defined in ops/control_flow_ops.py
    self._control_flow_context = None
    # A new node will depend of the union of all of the nodes in the stack.
    # In TF2.x or after switch_to_thread_local(),
    # self._thread_local._control_dependencies_stack is used instead.
    self._graph_control_dependencies_stack = []
    # Arbitrary collections of objects.
    self._collections = {}
    # The graph-level random seed
    self._seed = None
    # A dictionary of attributes that should be applied to all ops.
    self._attr_scope_map = {}
    # A map from op type to the kernel label that should be used.
    self._op_to_kernel_label_map = {}
    # A map from op type to an alternative op type that should be used when
    # computing gradients.
    self._gradient_override_map = {}
    # A map from op type to a gradient function that should be used instead.
    self._gradient_function_map = {}
    # True if the graph is considered "finalized".  In that case no
    # new operations can be added.
    self._finalized = False
    # Functions defined in the graph
    self._functions = collections.OrderedDict()
    # Default GraphDef versions
    self._graph_def_versions = versions_pb2.VersionDef(
        producer=versions.GRAPH_DEF_VERSION,
        min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER)
    self._building_function = False
    # Stack of colocate_with ops. In TF2.x or after switch_to_thread_local(),
    # self._thread_local._colocation_stack is used instead.
    self._graph_colocation_stack = traceable_stack.TraceableStack()
    # Set of tensors that are dangerous to feed!
    self._unfeedable_tensors = object_identity.ObjectIdentitySet()
    # Set of operations that are dangerous to fetch!
    self._unfetchable_ops = set()
    # A map of tensor handle placeholder to tensor dtype.
    self._handle_feeders = {}
    # A map from tensor handle to its read op.
    self._handle_readers = {}
    # A map from tensor handle to its move op.
    self._handle_movers = {}
    # A map from tensor handle to its delete op.
    self._handle_deleters = {}
    # Allow optimizers and other objects to pseudo-uniquely key graphs (this key
    # will be shared when defining function graphs, for example, so optimizers
    # being called inside function definitions behave as if they were seeing the
    # actual outside graph).
    self._graph_key = "graph-key-%d/" % (uid(),)
    # A string with the last reduction method passed to
    # losses.compute_weighted_loss(), or None. This is required only for
    # backward compatibility with Estimator and optimizer V1 use cases.
    self._last_loss_reduction = None
    # Flag that is used to indicate whether loss has been scaled by optimizer.
    # If this flag has been set, then estimator uses it to scale losss back
    # before reporting. This is required only for backward compatibility with
    # Estimator and optimizer V1 use cases.
    self._is_loss_scaled_by_optimizer = False
    self._container = ""
    # The current AutomaticControlDependencies context manager.
    self.experimental_acd_manager = None
    # Set to True if this graph is being built in an
    # AutomaticControlDependencies context.
    # Deprecated: use acd_manager instead.
    self._add_control_dependencies = False
    # Cache for OpDef protobufs retrieved via the C API.
    self._op_def_cache = {}
    # Cache for constant results of `broadcast_gradient_args()`. The keys are
    # tuples of fully-defined shapes: (x_shape_tuple, y_shape_tuple), and the
    # values are tuples of reduction indices: (rx, ry).
    self._bcast_grad_args_cache = {}
    # Cache for constant results of `reduced_shape()`. The keys are pairs of
    # tuples: (input_shape_tuple, reduction_indices_tuple), and the values
    # are pairs of tuples: (output_shape_kept_dims, tile_scaling).
    self._reduced_shape_cache = {}
    if tf2.enabled():
      self.switch_to_thread_local()
  # `Graph` now _is_ the C graph, but we have many places that manually attempt
  # to manipulate the _c_graph object. Leave these accessors here until these
  # are cleaned up.
  @property
  def _c_graph(self):
    return self
  def __enter__(self):
    return self
  def __exit__(self, *args):
    return
  def get(self):
    return self
  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @tf_contextlib.contextmanager
  def _variable_creator_scope(self, creator, priority=100):
    """Scope which defines a variable creation function.
    Args:
      creator: A callable taking `next_creator` and `kwargs`. See the
        `tf.variable_creator_scope` docstring.
      priority: Creators with a higher `priority` are called first. Within the
        same priority, creators are called inner-to-outer.
    Yields:
      `_variable_creator_scope` is a context manager with a side effect, but
      doesn't return a value.
    Raises:
      RuntimeError: If variable creator scopes are not properly nested.
    """
    # This step keeps a reference to the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = self._variable_creator_stack
    new = list(old)
    new.append((priority, creator))
    # Sorting is stable, so we'll put higher-priority creators later in the list
    # but otherwise maintain registration order.
    new.sort(key=lambda item: item[0])
    self._thread_local._variable_creator_stack = new  # pylint: disable=protected-access
    try:
      yield
    finally:
      if self._thread_local._variable_creator_stack is not new:  # pylint: disable=protected-access
        raise RuntimeError(
            "Exiting variable_creator_scope without proper nesting.")
      self._thread_local._variable_creator_stack = old  # pylint: disable=protected-access
  # TODO(b/192405401): unify resource_creator_scope with variable_creator_scope.
  # pylint: disable=protected-access
  @tf_contextlib.contextmanager
  def _resource_creator_scope(self, resource_type, creator):
    """Scope which defines a resource creation function used by some resource.
    The resource should be a subclass of CapturableResource with a class method
    `cls._resource_type`, the output of which is what the `resource_type`
    argument should be. By default, `cls._resource_type` returns the class name,
    `cls.__name__`. Given a scope, creators being added with the same
    `resource_type` argument will be composed together to apply to all classes
    with this `_resource_type`.
    `creator` is expected to be a function with the following signature:
    ```
      def resource_creator(next_creator, *a, **kwargs)
    ```
    The creator is supposed to eventually call the next_creator to create an
    instance if it does want to create an instance and not call
    the class initialization method directly. This helps make creators
    composable. A creator may choose to create multiple instances, return
    already existing instances, or simply register that an instance was created
    and defer to the next creator in line. Creators can also modify keyword
    arguments seen by the next creators.
    Valid keyword arguments in `kwargs` depends on the specific resource
    class. For StaticHashTable, this may be:
    * initializer: The table initializer to use.
    * default_value: The value to use if a key is missing in the table.
    * name: Optional name for the table, default to None.
    Args:
      resource_type: the output of the resource class's `_resource_type` method.
      creator: the passed creator for the resource.
    Yields:
      A scope in which the creator is active
    Raises:
      RuntimeError: If resource_creator_scope is existed without proper nesting.
    """
    # This step keeps a reference to the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = self._resource_creator_stack
    new = copy.deepcopy(old)
    if isinstance(resource_type, (list, tuple)):
      for r in resource_type:
        new[r].append(creator)
    else:
      new[resource_type].append(creator)
    self._thread_local._resource_creator_stack = new
    try:
      yield
    finally:
      if self._thread_local._resource_creator_stack is not new:
        raise RuntimeError(
            "Exiting resource_creator_scope without proper nesting.")
      self._thread_local._resource_creator_stack = old
  @property
  def _resource_creator_stack(self):
    if not hasattr(self._thread_local, "_resource_creator_stack"):
      self._thread_local._resource_creator_stack = collections.defaultdict(list)
    return self._thread_local._resource_creator_stack
  @_resource_creator_stack.setter
  def _resource_creator_stack(self, resource_creator_stack):
    self._thread_local._resource_creator_stack = resource_creator_stack
  # pylint: enable=protected-access
  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @property
  def _variable_creator_stack(self):
    if not hasattr(self._thread_local, "_variable_creator_stack"):
      self._thread_local._variable_creator_stack = []  # pylint: disable=protected-access
    # This previously returned a copy of the stack instead of the stack itself,
    # to guard against accidental mutation. Consider, however, code that wants
    # to save and restore the variable creator stack:
    #     def f():
    #       original_stack = graph._variable_creator_stack
    #       graph._variable_creator_stack = new_stack
    #       ...  # Some code
    #       graph._variable_creator_stack = original_stack
    #
    # And lets say you have some code that calls this function with some
    # variable_creator:
    #     def g():
    #       with variable_scope.variable_creator_scope(creator):
    #         f()
    # When exiting the variable creator scope, it would see a different stack
    # object than it expected leading to a "Exiting variable_creator_scope
    # without proper nesting" error.
    return self._thread_local._variable_creator_stack  # pylint: disable=protected-access
  @_variable_creator_stack.setter
  def _variable_creator_stack(self, variable_creator_stack):
    self._thread_local._variable_creator_stack = variable_creator_stack  # pylint: disable=protected-access
  def _check_not_finalized(self):
    """Check if the graph is finalized.
    Raises:
      RuntimeError: If the graph finalized.
    """
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")
  @property
  def graph_def_versions(self):
    # pylint: disable=line-too-long
    """The GraphDef version information of this graph.
    For details on the meaning of each version, see
    [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).
    Returns:
      A `VersionDef`.
    """
    return versions_pb2.VersionDef.FromString(self._version_def)
  @property
  def seed(self):
    """The graph-level random seed of this graph."""
    return self._seed
  @seed.setter
  def seed(self, seed):
    self._seed = seed
  @property
  def finalized(self):
    """True if this graph has been finalized."""
    return self._finalized
  def finalize(self):
    """Finalizes this graph, making it read-only.
    After calling `g.finalize()`, no new operations can be added to
    `g`.  This method is used to ensure that no operations are added
    to a graph when it is shared between multiple threads, for example
    when using a `tf.compat.v1.train.QueueRunner`.
    """
    self._finalized = True
  def _unsafe_unfinalize(self):
    """Opposite of `finalize`.
    Internal interface.
    NOTE: Unfinalizing a graph could have negative impact on performance,
    especially in a multi-threaded environment.  Unfinalizing a graph
    when it is in use by a Session may lead to undefined behavior. Ensure
    that all sessions using a graph are closed before calling this method.
    """
    self._finalized = False
  def _get_control_flow_context(self):
    """Returns the current control flow context.
    Returns:
      A context object.
    """
    return self._control_flow_context
  def _set_control_flow_context(self, ctx):
    """Sets the current control flow context.
    Args:
      ctx: a context object.
    """
    self._control_flow_context = ctx
  def _copy_functions_to_graph_def(self, graph_def, starting_bytesize):
    """If this graph contains functions, copy them to `graph_def`."""
    bytesize = starting_bytesize
    for f in self._functions.values():
      bytesize += f.cached_definition.ByteSize()
      if bytesize >= (1 << 31) or bytesize < 0:
        raise ValueError("GraphDef cannot be larger than 2GB.")
      graph_def.library.function.extend([f.cached_definition])
      if getattr(f, "grad_func_name", None):
        grad_def = function_pb2.GradientDef()
        grad_def.function_name = f.name
        grad_def.gradient_func = f.grad_func_name
        graph_def.library.gradient.extend([grad_def])
  def _as_graph_def(self, from_version=None, add_shapes=False):
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.
    The serialized `GraphDef` can be imported into another `Graph`
    (using `tf.import_graph_def`) or used with the
    [C++ Session API](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/api_docs/cc/index.md).
    This method is thread-safe.
    Args:
      from_version: Optional.  If this is set, returns a `GraphDef` containing
        only the nodes that were added to this graph since its `version`
        property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each node with
        the inferred shapes of each of its outputs.
    Returns:
      A tuple containing a
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer, and the version of the graph to which that
      `GraphDef` corresponds.
    Raises:
      ValueError: If the `graph_def` would be too large.
    """
    # pylint: enable=line-too-long
    with self._lock:
      with c_api_util.tf_buffer() as buf:
        with self._c_graph.get() as c_graph:
          pywrap_tf_session.TF_GraphToGraphDef(c_graph, buf)
          data = pywrap_tf_session.TF_GetBuffer(buf)
      graph = graph_pb2.GraphDef()
      graph.ParseFromString(compat.as_bytes(data))
      # Strip the experimental library field iff it's empty.
      if not graph.library.function:
        graph.ClearField("library")
      if add_shapes:
        for node in graph.node:
          op = self._get_operation_by_name(node.name)
          if op.outputs:
            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in op.outputs])
        for function_def in graph.library.function:
          defined_function = self._functions[function_def.signature.name]
          try:
            func_graph = defined_function.graph
          except AttributeError:
            # _DefinedFunction doesn't have a graph, _EagerDefinedFunction
            # does. Both rely on ops.py, so we can't really isinstance check
            # them.
            continue
          input_shapes = function_def.attr["_input_shapes"]
          try:
            func_graph_inputs = func_graph.inputs
          except AttributeError:
            continue
          # TODO(b/141471245): Fix the inconsistency when inputs of func graph
          # are appended during gradient computation of while/cond.
          assert len(input_shapes.list.shape) in [0, len(func_graph_inputs)]
          # If the function_def has inputs already filled out, skip this step.
          if not input_shapes.list.shape:
            for input_tensor, arg_def in zip(func_graph_inputs,
                                             function_def.signature.input_arg):
              input_shapes.list.shape.add().CopyFrom(
                  input_tensor.get_shape().as_proto())
              if input_tensor.dtype == dtypes.resource:
                _copy_handle_data_to_arg_def(input_tensor, arg_def)
          for output_tensor, arg_def in zip(func_graph.outputs,
                                            function_def.signature.output_arg):
            if output_tensor.dtype == dtypes.resource:
              _copy_handle_data_to_arg_def(output_tensor, arg_def)
          for node in function_def.node_def:
            try:
              op = func_graph.get_operation_by_name(node.name)
            except KeyError:
              continue
            outputs = op.outputs
            if op.type == "StatefulPartitionedCall":
              # Filter out any extra outputs (possibly added by function
              # backpropagation rewriting).
              num_outputs = len(node.attr["Tout"].list.type)
              outputs = outputs[:num_outputs]
            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in outputs])
    return graph, self.version
  def as_graph_def(self, from_version=None, add_shapes=False):
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.
    The serialized `GraphDef` can be imported into another `Graph`
    (using `tf.import_graph_def`) or used with the
    [C++ Session API](../../api_docs/cc/index.md).
    This method is thread-safe.
    Args:
      from_version: Optional.  If this is set, returns a `GraphDef` containing
        only the nodes that were added to this graph since its `version`
        property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each node with
        the inferred shapes of each of its outputs.
    Returns:
      A
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer.
    Raises:
      ValueError: If the `graph_def` would be too large.
    """
    # pylint: enable=line-too-long
    result, _ = self._as_graph_def(from_version, add_shapes)
    return result
  def _is_function(self, name):
    """Tests whether 'name' is registered in this graph's function library.
    Args:
      name: string op name.
    Returns:
      bool indicating whether or not 'name' is registered in function library.
    """
    return compat.as_str(name) in self._functions
  def _get_function(self, name):
    """Returns the function definition for 'name'.
    Args:
      name: string function name.
    Returns:
      The function def proto.
    """
    return self._functions.get(compat.as_str(name), None)
  def _add_function_recursive(self, function, overwrite=False):
    """Adds function to the graph including other functions in its graph."""
    if self._is_function(function.name):
      if overwrite:
        self._remove_function(function.name)
        self._add_function(function)
    else:
      self._add_function(function)
    if hasattr(function, "graph"):
      for f in function.graph._functions.values():  # pylint: disable=protected-access
        if self._is_function(f.name):
          if overwrite:
            self._remove_function(f.name)
            self._add_function(f)
        else:
          self._add_function(f)
  def _add_function(self, function):
    """Adds a function to the graph.
    After the function has been added, you can call to the function by
    passing the function name in place of an op name to
    `Graph.create_op()`.
    Args:
      function: A `_DefinedFunction` object.
    Raises:
      ValueError: if another function is defined with the same name.
    """
    self._check_not_finalized()
    name = function.name
    # Sanity checks on gradient definition for deprecated _DefinedFunction.
    if getattr(function, "grad_func_name", None) and getattr(
        function, "python_grad_func", None
    ):
      raise ValueError("Gradient defined twice for function %s" % name)
    # Add function to graph
    # pylint: disable=protected-access
    with self._c_graph.get() as c_graph:
      with function._c_func.get() as func:
        if getattr(function, "_grad_func", None):
          # For deprecated _DefinedFunction.
          with function._grad_func._c_func.get() as gradient:
            pywrap_tf_session.TF_GraphCopyFunction(c_graph, func, gradient)
        else:
          pywrap_tf_session.TF_GraphCopyFunction(c_graph, func, None)
    # pylint: enable=protected-access
    self._functions[compat.as_str(name)] = function
    # Need a new-enough consumer to support the functions we add to the graph.
    if self._graph_def_versions.min_consumer < 12:
      self._graph_def_versions.min_consumer = 12
  def _remove_function(self, name):
    self._check_not_finalized()
    if not self._is_function(name):
      raise ValueError(f"Function {name!r} is not found in {self!r}.")
    with self._c_graph.get() as c_graph:
      pywrap_tf_session.TF_GraphRemoveFunction(c_graph, compat.as_bytes(name))
      del self._functions[compat.as_str(name)]
  @property
  def building_function(self):
    """Returns True iff this graph represents a function."""
    return self._building_function
  # Helper functions to create operations.
  @deprecated_args(None,
                   "Shapes are always computed; don't use the compute_shapes "
                   "as it has no effect.", "compute_shapes")
  @traceback_utils.filter_traceback
  def create_op(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True):
    """Creates an `Operation` in this graph.
    This is a low-level interface for creating an `Operation`. Most
    programs will not call this method directly, and instead use the
    Python op constructors, such as `tf.constant()`, which add ops to
    the default graph.
    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) Deprecated. Has no effect (shapes are always
        computed).
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.
    Raises:
      TypeError: if any of the inputs is not a `Tensor`.
      ValueError: if colocation conflicts with existing device assignment.
    Returns:
      An `Operation` object.
    """
    del compute_shapes
    for idx, a in enumerate(inputs):
      if not isinstance(a, Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    return self._create_op_internal(op_type, inputs, dtypes, input_types, name,
                                    attrs, op_def, compute_device)
  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    """Creates an `Operation` in this graph.
    Implements `Graph.create_op()` without the overhead of the deprecation
    wrapper.
    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.
    Raises:
      ValueError: if colocation conflicts with existing device assignment.
    Returns:
      An `Operation` object.
    """
    self._check_not_finalized()
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = name_from_scope_name(name)
    else:
      name = self.unique_name(name)
    node_def = _NodeDef(op_type, name, attrs)
    input_ops = set(t.op for t in inputs)
    control_inputs = self._control_dependencies_for_inputs(input_ops)
    # _create_op_helper mutates the new Operation. `_mutation_lock` ensures a
    # Session.run call cannot occur between creating and mutating the op.
    with self._mutation_lock():
      ret = Operation.from_node_def(
          node_def,
          self,
          inputs=inputs,
          output_types=dtypes,
          control_inputs=control_inputs,
          input_types=input_types,
          original_op=self._default_original_op,
          op_def=op_def,
      )
      self._create_op_helper(ret, compute_device=compute_device)
    return ret
  def _create_op_from_tf_operation(self, c_op, compute_device=True):
    """Creates an `Operation` in this graph from the supplied TF_Operation.
    This method is like create_op() except the new Operation is constructed
    using `c_op`. The returned Operation will have `c_op` as its _c_op
    field. This is used to create Operation objects around TF_Operations created
    indirectly by the C API (e.g. by TF_ImportGraphDef, TF_FinishWhile).
    This function does not call Operation._control_flow_post_processing or
    Graph._control_dependencies_for_inputs (since the inputs may not be
    available yet). The caller is responsible for calling these methods.
    Args:
      c_op: a wrapped TF_Operation
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.
    Returns:
      An `Operation` object.
    """
    self._check_not_finalized()
    ret = Operation._from_c_op(c_op=c_op, g=self)  # pylint: disable=protected-access
    # If a name_scope was created with ret.name but no nodes were created in it,
    # the name will still appear in _names_in_use even though the name hasn't
    # been used. This is ok, just leave _names_in_use as-is in this case.
    # TODO(skyewm): make the C API guarantee no name conflicts.
    name_key = ret.name.lower()
    if name_key not in self._names_in_use:
      self._names_in_use[name_key] = 1
    self._create_op_helper(ret, compute_device=compute_device)
    return ret
  def _create_op_helper(self, op, compute_device=True):
    """Common logic for creating an op in this graph."""
    # Apply any additional attributes requested. Do not overwrite any existing
    # attributes.
    for key, value in self._attr_scope_map.items():
      try:
        op.get_attr(key)
      except ValueError:
        if callable(value):
          value = value(op.node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" %
                (key, value))
        if value:
          op._set_attr(key, value)  # pylint: disable=protected-access
    # Apply a kernel label if one has been specified for this op type.
    try:
      kernel_label = self._op_to_kernel_label_map[op.type]
      op._set_attr("_kernel",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass
    op._gradient_function = self._gradient_function_map.get(op.type)  # pylint: disable=protected-access
    # Apply the overriding op type for gradients if one has been specified for
    # this op type.
    try:
      mapped_op_type = self._gradient_override_map[op.type]
      op._set_attr("_gradient_op_type",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass
    self._record_op_seen_by_control_dependencies(op)
    if compute_device:
      self._apply_device_functions(op)
    # Snapshot the colocation stack metadata before we might generate error
    # messages using it.  Note that this snapshot depends on the actual stack
    # and is independent of the op's _class attribute.
    # pylint: disable=protected-access
    op._colocation_code_locations = self._snapshot_colocation_stack_metadata()
    # pylint: enable=protected-access
    if self._colocation_stack:
      all_colocation_groups = []
      is_device_set = False
      for colocation_op in self._colocation_stack.peek_objs():
        try:
          all_colocation_groups.extend(colocation_op.colocation_groups())
        except AttributeError:
          pass
        if colocation_op.device and not is_device_set:
          # pylint: disable=protected-access
          op._set_device(colocation_op.device)
          # pylint: enable=protected-access
          is_device_set = True
      all_colocation_groups = sorted(set(all_colocation_groups))
      # pylint: disable=protected-access
      op._set_attr(
          "_class",
          attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
      # pylint: enable=protected-access
    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if self._container and op._is_stateful:  # pylint: disable=protected-access
      try:
        container_attr = op.get_attr("container")
      except ValueError:
        # "container" attribute is not in OpDef
        pass
      else:
        if not container_attr:
          op._set_attr("container", attr_value_pb2.AttrValue(  # pylint: disable=protected-access
              s=compat.as_bytes(self._container)))
  def _add_new_tf_operations(self, compute_devices=True):
    """Creates `Operations` in this graph for any new TF_Operations.
    This is useful for when TF_Operations are indirectly created by the C API
    outside of the Operation constructor (e.g. by TF_ImportGraphDef,
    TF_FinishWhile). This ensures there are corresponding Operations for all
    TF_Operations in the underlying TF_Graph.
    Args:
      compute_devices: (Optional.) If True, device functions will be executed to
        compute the device properties of each new Operation.
    Returns:
      A list of the new `Operation` objects.
    """
    self._check_not_finalized()
    # Create all Operation objects before accessing their inputs since an op may
    # be created before its inputs.
    new_ops = [
        self._create_op_from_tf_operation(c_op, compute_device=compute_devices)
        for c_op in self.new_operations()
    ]
    # pylint: disable=protected-access
    for op in new_ops:
      new_control_inputs = self._control_dependencies_for_inputs(op.inputs)
      op._add_control_inputs(new_control_inputs)
      op._control_flow_post_processing()
    # pylint: enable=protected-access
    return new_ops
  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    """Returns the object referred to by `obj`, as an `Operation` or `Tensor`.
    This function validates that `obj` represents an element of this
    graph, and gives an informative error message if it is not.
    This function is the canonical way to get/validate an object of
    one of the allowed types from an external argument reference in the
    Session API.
    This method may be called concurrently from multiple threads.
    Args:
      obj: A `Tensor`, an `Operation`, or the name of a tensor or operation. Can
        also be any object with an `_as_graph_element()` method that returns a
        value of one of these types. Note: `_as_graph_element` will be called
        inside the graph's lock and so may not modify the graph.
      allow_tensor: If true, `obj` may refer to a `Tensor`.
      allow_operation: If true, `obj` may refer to an `Operation`.
    Returns:
      The `Tensor` or `Operation` in the Graph corresponding to `obj`.
    Raises:
      TypeError: If `obj` is not a type we support attempting to convert
        to types.
      ValueError: If `obj` is of an appropriate type but invalid. For
        example, an invalid string.
      KeyError: If `obj` is not an object in the graph.
    """
    if self._finalized:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)
    with self._lock:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)
  def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):
    """See `Graph.as_graph_element()` for details."""
    # The vast majority of this function is figuring
    # out what an API user might be doing wrong, so
    # that we can give helpful error messages.
    #
    # Ideally, it would be nice to split it up, but we
    # need context to generate nice error messages.
    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")
    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:
      obj = temp_obj
    # If obj appears to be a name...
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)
      if ":" in name and allow_tensor:
        # Looks like a Tensor name and can be a Tensor.
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        try:
          op = self._get_operation_by_name(op_name)
        except KeyError as exc:
          raise KeyError(
              "The name %s refers to a Tensor which does not "
              "exist. The operation, %s, does not exist in the "
              "graph." % (repr(name), repr(op_name))
          ) from exc
        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs." %
                         (repr(name), repr(op_name), len(op.outputs)))
      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s." %
                         (repr(name), types_str))
      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        try:
          op = self._get_operation_by_name(name)
        except KeyError as exc:
          raise KeyError(
              "The name %s refers to an Operation not in the graph."
              % repr(name)
          ) from exc
        return op
      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        try:
          op = self._get_operation_by_name(name)
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s." %
                     (repr(name), types_str))
        except KeyError:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)
    elif isinstance(obj, Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s." %
                      (type(obj).__name__, types_str))
  def get_operation_by_name(self, name):
    """Returns the `Operation` with the given `name`.
    This method may be called concurrently from multiple threads.
    Args:
      name: The name of the `Operation` to return.
    Returns:
      The `Operation` with the given `name`.
    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to an operation in this graph.
    """
    if not isinstance(name, str):
      raise TypeError("Operation names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=False, allow_operation=True)
  def _get_operation_by_tf_operation(self, tf_oper):
    op_name = pywrap_tf_session.TF_OperationName(tf_oper)
    return self._get_operation_by_name(op_name)
  def get_tensor_by_name(self, name):
    """Returns the `Tensor` with the given `name`.
    This method may be called concurrently from multiple threads.
    Args:
      name: The name of the `Tensor` to return.
    Returns:
      The `Tensor` with the given `name`.
    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to a tensor in this graph.
    """
    # Names should be strings.
    if not isinstance(name, str):
      raise TypeError("Tensor names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=True, allow_operation=False)
  def _get_tensor_by_tf_output(self, tf_output):
    """Returns the `Tensor` representing `tf_output`.
    Note that there is only one such `Tensor`, i.e. multiple calls to this
    function with the same TF_Output value will always return the same `Tensor`
    object.
    Args:
      tf_output: A wrapped `TF_Output` (the C API equivalent of `Tensor`).
    Returns:
      The `Tensor` that represents `tf_output`.
    """
    op = self._get_operation_by_tf_operation(tf_output.oper)
    return op.outputs[tf_output.index]
  def op_def_for_type(self, type):  # pylint: disable=redefined-builtin
    """Returns the `OpDef` proto for `type`. `type` is a string."""
    # NOTE: No locking is required because the lookup and insertion operations
    # on Python dictionaries are atomic.
    try:
      return self._op_def_cache[type]
    except KeyError:
      self._op_def_cache[type] = op_def_pb2.OpDef.FromString(
          self._op_def_for_type(type)
      )
      return self._op_def_cache[type]
  def as_default(self):
    """Returns a context manager that makes this `Graph` the default graph.
    This method should be used if you want to create multiple graphs
    in the same process. For convenience, a global default graph is
    provided, and all ops will be added to this graph if you do not
    create a new graph explicitly.
    Use this method with the `with` keyword to specify that ops created within
    the scope of a block should be added to this graph. In this case, once
    the scope of the `with` is exited, the previous default graph is set again
    as default. There is a stack, so it's ok to have multiple nested levels
    of `as_default` calls.
    The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default graph in that
    thread, you must explicitly add a `with g.as_default():` in that
    thread's function.
    The following code examples are equivalent:
    ```python
    # 1. Using Graph.as_default():
    g = tf.Graph()
    with g.as_default():
      c = tf.constant(5.0)
      assert c.graph is g
    # 2. Constructing and making default:
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      assert c.graph is g
    ```
    If eager execution is enabled ops created under this context manager will be
    added to the graph instead of executed eagerly.
    Returns:
      A context manager for using this graph as the default graph.
    """
    return _default_graph_stack.get_controller(self)
  @property
  def collections(self):
    """Returns the names of the collections known to this graph."""
    return list(self._collections)
  def add_to_collection(self, name, value):
    """Stores `value` in the collection with the given `name`.
    Note that collections are not sets, so it is possible to add a value to
    a collection several times.
    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
      value: The value to add to the collection.
    """  # pylint: disable=g-doc-exception
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)
  def add_to_collections(self, names, value):
    """Stores `value` in the collections given by `names`.
    Note that collections are not sets, so it is possible to add a value to
    a collection several times. This function makes sure that duplicates in
    `names` are ignored, but it will not check for pre-existing membership of
    `value` in any of the collections in `names`.
    `names` can be any iterable, but if `names` is a string, it is treated as a
    single collection name.
    Args:
      names: The keys for the collections to add to. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collections.
    """
    # Make sure names are unique, but treat strings as a single collection name
    names = (names,) if isinstance(names, str) else set(names)
    for name in names:
      self.add_to_collection(name, value)
  def get_collection_ref(self, name):
    """Returns a list of values in the collection with the given `name`.
    If the collection exists, this returns the list itself, which can
    be modified in place to change the collection.  If the collection does
    not exist, it is created as an empty list and the list is returned.
    This is different from `get_collection()` which always returns a copy of
    the collection list if it exists and never creates an empty collection.
    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
    Returns:
      The list of values in the collection with the given `name`, or an empty
      list if no value has been added to that collection.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list
  def get_collection(self, name, scope=None):
    """Returns a list of values in the collection with the given `name`.
    This is different from `get_collection_ref()` which always returns the
    actual collection list if it exists in that it returns a new list each time
    it is called.
    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
      scope: (Optional.) A string. If supplied, the resulting list is filtered
        to include only items whose `name` attribute matches `scope` using
        `re.match`. Items without a `name` attribute are never returned if a
        scope is supplied. The choice of `re.match` means that a `scope` without
        special tokens filters by prefix.
    Returns:
      The list of values in the collection with the given `name`, or
      an empty list if no value has been added to that collection. The
      list contains the values in the order under which they were
      collected.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      collection = self._collections.get(name, None)
      if collection is None:
        return []
      if scope is None:
        return list(collection)
      else:
        c = []
        regex = re.compile(scope)
        for item in collection:
          try:
            if regex.match(item.name):
              c.append(item)
          except AttributeError:
            # Collection items with no name are ignored.
            pass
        return c
  def get_all_collection_keys(self):
    """Returns a list of collections used in this graph."""
    with self._lock:
      return [x for x in self._collections if isinstance(x, str)]
  def clear_collection(self, name):
    """Clears all values in a collection.
    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
    """
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]
  @tf_contextlib.contextmanager
  def _original_op(self, op):
    """Python 'with' handler to help annotate ops with their originator.
    An op may have an 'original_op' property that indicates the op on which
    it was based. For example a replica op is based on the op that was
    replicated and a gradient op is based on the op that was differentiated.
    All ops created in the scope of this 'with' handler will have
    the given 'op' as their original op.
    Args:
      op: The Operation that all ops created in this scope will have as their
        original op.
    Yields:
      Nothing.
    """
    old_original_op = self._default_original_op
    self._default_original_op = op
    try:
      yield
    finally:
      self._default_original_op = old_original_op
  @property
  def _name_stack(self):
    # This may be called from a thread where name_stack doesn't yet exist.
    if not hasattr(self._thread_local, "_name_stack"):
      self._thread_local._name_stack = ""
    return self._thread_local._name_stack
  @_name_stack.setter
  def _name_stack(self, name_stack):
    self._thread_local._name_stack = name_stack
  # pylint: disable=g-doc-return-or-yield,line-too-long
  @tf_contextlib.contextmanager
  def name_scope(self, name):
    """Returns a context manager that creates hierarchical names for operations.
    A graph maintains a stack of name scopes. A `with name_scope(...):`
    statement pushes a new name onto the stack for the lifetime of the context.
    The `name` argument will be interpreted as follows:
    * A string (not ending with '/') will create a new name scope, in which
      `name` is appended to the prefix of all operations created in the
      context. If `name` has been used before, it will be made unique by
      calling `self.unique_name(name)`.
    * A scope previously captured from a `with g.name_scope(...) as
      scope:` statement will be treated as an "absolute" name scope, which
      makes it possible to re-enter existing scopes.
    * A value of `None` or the empty string will reset the current name scope
      to the top-level (empty) name scope.
    For example:
    ```python
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, name="c")
      assert c.op.name == "c"
      c_1 = tf.constant(6.0, name="c")
      assert c_1.op.name == "c_1"
      # Creates a scope called "nested"
      with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"
        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
          nested_inner_c = tf.constant(20.0, name="c")
          assert nested_inner_c.op.name == "nested/inner/c"
        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
          nested_inner_1_c = tf.constant(30.0, name="c")
          assert nested_inner_1_c.op.name == "nested/inner_1/c"
          # Treats `scope` as an absolute name scope, and
          # switches to the "nested/" scope.
          with g.name_scope(scope):
            nested_d = tf.constant(40.0, name="d")
            assert nested_d.op.name == "nested/d"
            with g.name_scope(""):
              e = tf.constant(50.0, name="e")
              assert e.op.name == "e"
    ```
    The name of the scope itself can be captured by `with
    g.name_scope(...) as scope:`, which stores the name of the scope
    in the variable `scope`. This value can be used to name an
    operation that represents the overall result of executing the ops
    in a scope. For example:
    ```python
    inputs = tf.constant(...)
    with g.name_scope('my_layer') as scope:
      weights = tf.Variable(..., name="weights")
      biases = tf.Variable(..., name="biases")
      affine = tf.matmul(inputs, weights) + biases
      output = tf.nn.relu(affine, name=scope)
    ```
    NOTE: This constructor validates the given `name`. Valid scope
    names match one of the following regular expressions:
        [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
        [A-Za-z0-9_.\\-/]* (for other scopes)
    Args:
      name: A name for the scope.
    Returns:
      A context manager that installs `name` as a new name scope.
    Raises:
      ValueError: If `name` is not a valid scope name, according to the rules
        above.
    """
    if name:
      if isinstance(name, compat.bytes_or_text_types):
        name = compat.as_str(name)
      if self._name_stack:
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError(
              f"'{name}' is not a valid scope name. A scope name has to match "
              f"the following pattern: {_VALID_SCOPE_NAME_REGEX.pattern}")
      else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError(
              f"'{name}' is not a valid root scope name. A root scope name has "
              f"to match the following pattern: {_VALID_OP_NAME_REGEX.pattern}")
    old_stack = self._name_stack
    if not name:  # Both for name=None and name="" we re-set to empty scope.
      new_stack = ""
      returned_scope = ""
    elif name[-1] == "/":
      new_stack = name_from_scope_name(name)
      returned_scope = name
    else:
      new_stack = self.unique_name(name)
      returned_scope = new_stack + "/"
    self._name_stack = new_stack
    try:
      yield returned_scope
    finally:
      self._name_stack = old_stack
  # pylint: enable=g-doc-return-or-yield,line-too-long
  def unique_name(self, name, mark_as_used=True):
    """Return a unique operation name for `name`.
    Note: You rarely need to call `unique_name()` directly.  Most of
    the time you just need to create `with g.name_scope()` blocks to
    generate structured names.
    `unique_name` is used to generate structured names, separated by
    `"/"`, to help identify operations when debugging a graph.
    Operation names are displayed in error messages reported by the
    TensorFlow runtime, and in various visualization tools such as
    TensorBoard.
    If `mark_as_used` is set to `True`, which is the default, a new
    unique name is created and marked as in use. If it's set to `False`,
    the unique name is returned without actually being marked as used.
    This is useful when the caller simply wants to know what the name
    to be created will be.
    Args:
      name: The name for an operation.
      mark_as_used: Whether to mark this name as being used.
    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    if self._name_stack:
      name = self._name_stack + "/" + name
    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    # Increment the number for "name_key".
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      # Make sure the composed name key is not already used.
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      # Mark the composed name_key as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name_key] = 1
      # Return the new name with the original capitalization of the given name.
      name = "%s_%d" % (name, i - 1)
    return name
  def get_name_scope(self):
    """Returns the current name scope.
    For example:
    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.compat.v1.get_default_graph().get_name_scope())
    ```
    would print the string `scope1/scope2`.
    Returns:
      A string representing the current name scope.
    """
    return self._name_stack
  @tf_contextlib.contextmanager
  def _colocate_with_for_gradient(self, op, gradient_uid,
                                  ignore_existing=False):
    with self.colocate_with(op, ignore_existing):
      if gradient_uid is not None:
        ctx = _get_enclosing_context(self)
        if ctx is not None:
          ctx.EnterGradientColocation(op, gradient_uid)
          try:
            yield
          finally:
            ctx.ExitGradientColocation(op, gradient_uid)
        else:
          yield
      else:
        yield
  @tf_contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False):
    """Returns a context manager that specifies an op to colocate with.
    Note: this function is not for public use, only for internal libraries.
    For example:
    ```python
    a = tf.Variable([1.0])
    with g.colocate_with(a):
      b = tf.constant(1.0)
      c = tf.add(a, b)
    ```
    `b` and `c` will always be colocated with `a`, no matter where `a`
    is eventually placed.
    **NOTE** Using a colocation scope resets any existing device constraints.
    If `op` is `None` then `ignore_existing` must be `True` and the new
    scope resets all colocation and device constraints.
    Args:
      op: The op to colocate all created ops with, or `None`.
      ignore_existing: If true, only applies colocation of this op within the
        context, rather than applying all colocation properties on the stack.
        If `op` is `None`, this value must be `True`.
    Raises:
      ValueError: if op is None but ignore_existing is False.
    Yields:
      A context manager that specifies the op with which to colocate
      newly created ops.
    """
    if op is None and not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")
    op, device_only_candidate = _op_to_colocate_with(op, self)
    # By default, colocate_with resets the device function stack,
    # since colocate_with is typically used in specific internal
    # library functions where colocation is intended to be "stronger"
    # than device functions.
    #
    # In the future, a caller may specify that device_functions win
    # over colocation, in which case we can add support.
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = traceable_stack.TraceableStack()
    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = traceable_stack.TraceableStack()
    if op is not None:
      # offset refers to the stack frame used for storing code location.
      # We use 4, the sum of 1 to use our caller's stack frame and 3
      # to jump over layers of context managers above us.
      self._colocation_stack.push_obj(op, offset=4)
      if device_only_candidate is not None:
        self._colocation_stack.push_obj(device_only_candidate, offset=4)
    elif not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")
    try:
      yield
    finally:
      # Restore device function stack
      self._device_function_stack = device_fn_tmp
      if op is not None:
        self._colocation_stack.pop_obj()
        if device_only_candidate is not None:
          self._colocation_stack.pop_obj()
      # Reset the colocation stack if requested.
      if ignore_existing:
        self._colocation_stack = current_stack
  def _add_device_to_stack(self, device_name_or_function, offset=0):
    """Add device to stack manually, separate from a context manager."""
    total_offset = 1 + offset
    spec = _UserDeviceSpec(device_name_or_function)
    self._device_function_stack.push_obj(spec, offset=total_offset)
    return spec
  @tf_contextlib.contextmanager
  def device(self, device_name_or_function):
    # pylint: disable=line-too-long
    """Returns a context manager that specifies the default device to use.
    The `device_name_or_function` argument may either be a device name
    string, a device function, or None:
    * If it is a device name string, all operations constructed in
      this context will be assigned to the device with that name, unless
      overridden by a nested `device()` context.
    * If it is a function, it will be treated as a function from
      Operation objects to device name strings, and invoked each time
      a new Operation is created. The Operation will be assigned to
      the device with the returned name.
    * If it is None, all `device()` invocations from the enclosing context
      will be ignored.
    For information about the valid syntax of device name strings, see
    the documentation in
    [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
    For example:
    ```python
    with g.device('/device:GPU:0'):
      # All operations constructed in this context will be placed
      # on GPU 0.
      with g.device(None):
        # All operations constructed in this context will have no
        # assigned device.
    # Defines a function from `Operation` to device string.
    def matmul_on_gpu(n):
      if n.type == "MatMul":
        return "/device:GPU:0"
      else:
        return "/cpu:0"
    with g.device(matmul_on_gpu):
      # All operations of type "MatMul" constructed in this context
      # will be placed on GPU 0; all other operations will be placed
      # on CPU 0.
    ```
    **N.B.** The device scope may be overridden by op wrappers or
    other library code. For example, a variable assignment op
    `v.assign()` must be colocated with the `tf.Variable` `v`, and
    incompatible device scopes will be ignored.
    Args:
      device_name_or_function: The device name or function to use in the
        context.
    Yields:
      A context manager that specifies the default device to use for newly
      created ops.
    Raises:
      RuntimeError: If device scopes are not properly nested.
    """
    self._add_device_to_stack(device_name_or_function, offset=2)
    old_top_of_stack = self._device_function_stack.peek_top_obj()
    try:
      yield
    finally:
      new_top_of_stack = self._device_function_stack.peek_top_obj()
      if old_top_of_stack is not new_top_of_stack:
        raise RuntimeError("Exiting device scope without proper scope nesting.")
      self._device_function_stack.pop_obj()
  def _apply_device_functions(self, op):
    """Applies the current device function stack to the given operation."""
    # Apply any device functions in LIFO order, so that the most recently
    # pushed function has the first chance to apply a device to the op.
    # We apply here because the result can depend on the Operation's
    # signature, which is computed in the Operation constructor.
    # pylint: disable=protected-access
    prior_device_string = None
    for device_spec in self._device_function_stack.peek_objs():
      if device_spec.is_null_merge:
        continue
      if device_spec.function is None:
        break
      device_string = device_spec.string_merge(op)
      # Take advantage of the fact that None is a singleton and Python interns
      # strings, since identity checks are faster than equality checks.
      if device_string is not prior_device_string:
        op._set_device_from_string(device_string)
        prior_device_string = device_string
    op._device_code_locations = self._snapshot_device_function_stack_metadata()
    # pylint: enable=protected-access
  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def container(self, container_name):
    """Returns a context manager that specifies the resource container to use.
    Stateful operations, such as variables and queues, can maintain their
    states on devices so that they can be shared by multiple processes.
    A resource container is a string name under which these stateful
    operations are tracked. These resources can be released or cleared
    with `tf.Session.reset()`.
    For example:
    ```python
    with g.container('experiment0'):
      # All stateful Operations constructed in this context will be placed
      # in resource container "experiment0".
      v1 = tf.Variable([1.0])
      v2 = tf.Variable([2.0])
      with g.container("experiment1"):
        # All stateful Operations constructed in this context will be
        # placed in resource container "experiment1".
        v3 = tf.Variable([3.0])
        q1 = tf.queue.FIFOQueue(10, tf.float32)
      # All stateful Operations constructed in this context will be
      # be created in the "experiment0".
      v4 = tf.Variable([4.0])
      q1 = tf.queue.FIFOQueue(20, tf.float32)
      with g.container(""):
        # All stateful Operations constructed in this context will be
        # be placed in the default resource container.
        v5 = tf.Variable([5.0])
        q3 = tf.queue.FIFOQueue(30, tf.float32)
    # Resets container "experiment0", after which the state of v1, v2, v4, q1
    # will become undefined (such as uninitialized).
    tf.Session.reset(target, ["experiment0"])
    ```
    Args:
      container_name: container name string.
    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    self._container = container_name
    try:
      yield self._container
    finally:
      self._container = original_container
  # pylint: enable=g-doc-return-or-yield
  class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""
    def __init__(self, graph, control_inputs):
      """Create a new `_ControlDependenciesController`.
      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.
      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.
      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.
      Args:
        graph: The graph that this controller is managing.
        control_inputs: List of ops to use as control inputs in addition to the
          current control dependencies.  None to indicate that the dependencies
          should be cleared.
      """
      self._graph = graph
      if control_inputs is None:
        self._control_inputs_val = []
        self._new_stack = True
      else:
        self._control_inputs_val = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None
    # pylint: disable=protected-access
    def __enter__(self):
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)
    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)
    # pylint: enable=protected-access
    @property
    def control_inputs(self):
      return self._control_inputs_val
    def add_op(self, op):
      if isinstance(op, Tensor):
        op = op.ref()
      self._seen_nodes.add(op)
    def op_in_group(self, op):
      if isinstance(op, Tensor):
        op = op.ref()
      return op in self._seen_nodes
  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)
  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()
  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret
  def _control_dependencies_for_inputs(self, input_ops):
    """For an op that takes `input_ops` as inputs, compute control inputs.
    The returned control dependencies should yield an execution that
    is equivalent to adding all control inputs in
    self._control_dependencies_stack to a newly created op. However,
    this function attempts to prune the returned control dependencies
    by observing that nodes created within the same `with
    control_dependencies(...):` block may have data dependencies that make
    the explicit approach redundant.
    Args:
      input_ops: The data input ops for an op to be created.
    Returns:
      A list of control inputs for the op to be created.
    """
    ret = []
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependencies for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend(c for c in controller.control_inputs if c not in input_ops)
    return ret
  def _record_op_seen_by_control_dependencies(self, op):
    """Record that the given op depends on all registered control dependencies.
    Args:
      op: An Operation.
    """
    for controller in self._control_dependencies_stack:
      controller.add_op(op)
  def control_dependencies(self, control_inputs):
    """Returns a context manager that specifies control dependencies.
    Use with the `with` keyword to specify that all operations constructed
    within the context should have control dependencies on
    `control_inputs`. For example:
    ```python
    with g.control_dependencies([a, b, c]):
      # `d` and `e` will only run after `a`, `b`, and `c` have executed.
      d = ...
      e = ...
    ```
    Multiple calls to `control_dependencies()` can be nested, and in
    that case a new `Operation` will have control dependencies on the union
    of `control_inputs` from all active contexts.
    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies([c, d]):
        # Ops constructed here run after `a`, `b`, `c`, and `d`.
    ```
    You can pass None to clear the control dependencies:
    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies(None):
        # Ops constructed here run normally, not waiting for either `a` or `b`.
        with g.control_dependencies([c, d]):
          # Ops constructed here run after `c` and `d`, also not waiting
          # for either `a` or `b`.
    ```
    *N.B.* The control dependencies context applies *only* to ops that
    are constructed within the context. Merely using an op or tensor
    in the context does not add a control dependency. The following
    example illustrates this point:
    ```python
    # WRONG
    def my_func(pred, tensor):
      t = tf.matmul(tensor, tensor)
      with tf.control_dependencies([pred]):
        # The matmul op is created outside the context, so no control
        # dependency will be added.
        return t
    # RIGHT
    def my_func(pred, tensor):
      with tf.control_dependencies([pred]):
        # The matmul op is created in the context, so a control dependency
        # will be added.
        return tf.matmul(tensor, tensor)
    ```
    Also note that though execution of ops created under this scope will trigger
    execution of the dependencies, the ops created under this scope might still
    be pruned from a normal tensorflow graph. For example, in the following
    snippet of code the dependencies are never executed:
    ```python
      loss = model.loss()
      with tf.control_dependencies(dependencies):
        loss = loss + tf.constant(1)  # note: dependencies ignored in the
                                      # backward pass
      return tf.gradients(loss, model.variables)
    ```
    This is because evaluating the gradient graph does not require evaluating
    the constant(1) op created in the forward pass.
    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which must be
        executed or computed before running the operations defined in the
        context.  Can also be `None` to clear the control dependencies.
    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.
    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      # The hasattr(handle) is designed to match ResourceVariables. This is so
      # control dependencies on a variable or on an unread variable don't
      # trigger reads.
      if (isinstance(c, internal.IndexedSlices) or
          (hasattr(c, "_handle") and hasattr(c, "op"))):
        c = c.op
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
        # Mark this op with an attribute indicating that it is used as a manual
        # control dep in order to allow tracking how common utilization of
        # manual control deps in graphs run through the MLIR Bridge are. See
        # go/manual-control-dependencies-bridge for details.
        # pylint: disable=protected-access
        c._set_attr("_has_manual_control_dependencies",
                    attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access
    return self._ControlDependenciesController(self, control_ops)
  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _attr_scope(self, attr_map):
    """EXPERIMENTAL: A context manager for setting attributes on operators.
    This context manager can be used to add additional
    attributes to operators within the scope of the context.
    For example:
       with ops.Graph().as_default() as g:
         f_1 = Foo()  # No extra attributes
         with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=False)}):
           f_2 = Foo()  # Additional attribute _a=False
           with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=True)}):
             f_3 = Foo()  # Additional attribute _a=False
             with g._attr_scope({"_a": None}):
               f_4 = Foo()  # No additional attributes.
    Args:
      attr_map: A dictionary mapping attr name strings to AttrValue protocol
        buffers or None.
    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.
    Raises:
      TypeError: If attr_map is not a dictionary mapping
        strings to AttrValue protobufs.
    """
    if not isinstance(attr_map, dict):
      raise TypeError("attr_map must be a dictionary mapping "
                      "strings to AttrValue protocol buffers")
    # The saved_attrs dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_attrs = {}
    # Install the given attribute
    for name, attr in attr_map.items():
      if not (isinstance(name, str) and
              (isinstance(attr, (type(None), attr_value_pb2.AttrValue)) or
               callable(attr))):
        raise TypeError("attr_map must be a dictionary mapping "
                        "strings to AttrValue protocol buffers or "
                        "callables that emit AttrValue protocol buffers")
      try:
        saved_attrs[name] = self._attr_scope_map[name]
      except KeyError:
        pass
      if attr is None:
        del self._attr_scope_map[name]
      else:
        self._attr_scope_map[name] = attr
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the attributes set for this context, and restore any saved
      # attributes.
      for name, attr in attr_map.items():
        try:
          self._attr_scope_map[name] = saved_attrs[name]
        except KeyError:
          del self._attr_scope_map[name]
  # pylint: enable=g-doc-return-or-yield
  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map):
    """EXPERIMENTAL: A context manager for setting kernel labels.
    This context manager can be used to select particular
    implementations of kernels within the scope of the context.
    For example:
        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.
    Args:
      op_to_kernel_label_map: A dictionary mapping op type strings to kernel
        label strings.
    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.
    Raises:
      TypeError: If op_to_kernel_label_map is not a dictionary mapping
        strings to strings.
    """
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, str) and
              isinstance(label, str)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]
  # pylint: enable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _override_gradient_function(self, gradient_function_map):
    """Specify gradient function for the given op type."""
    # This is an internal API and we don't need nested context for this.
    # TODO(mdan): make it a proper context manager.
    assert not self._gradient_function_map
    self._gradient_function_map = gradient_function_map
    try:
      yield
    finally:
      self._gradient_function_map = {}
  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def gradient_override_map(self, op_type_map):
    """EXPERIMENTAL: A context manager for overriding gradient functions.
    This context manager can be used to override the gradient function
    that will be used for ops within the scope of the context.
    For example:
    ```python
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      # ...
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                              # gradient of s_2.
    ```
    Args:
      op_type_map: A dictionary mapping op type strings to alternative op type
        strings.
    Returns:
      A context manager that sets the alternative op type to be used for one
      or more ops created in that context.
    Raises:
      TypeError: If `op_type_map` is not a dictionary mapping strings to
        strings.
    """
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, str) and
              isinstance(mapped_op_type, str)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]
  # pylint: enable=g-doc-return-or-yield
  def prevent_feeding(self, tensor):
    """Marks the given `tensor` as unfeedable in this graph."""
    self._unfeedable_tensors.add(tensor)
  def is_feedable(self, tensor):
    """Returns `True` if and only if `tensor` is feedable."""
    return tensor not in self._unfeedable_tensors
  def prevent_fetching(self, op):
    """Marks the given `op` as unfetchable in this graph."""
    self._unfetchable_ops.add(op)
  def is_fetchable(self, tensor_or_op):
    """Returns `True` if and only if `tensor_or_op` is fetchable."""
    if isinstance(tensor_or_op, Tensor):
      return tensor_or_op.op not in self._unfetchable_ops
    else:
      return tensor_or_op not in self._unfetchable_ops
  def switch_to_thread_local(self):
    """Make device, colocation and dependencies stacks thread-local.
    Device, colocation and dependencies stacks are not thread-local be default.
    If multiple threads access them, then the state is shared.  This means that
    one thread may affect the behavior of another thread.
    After this method is called, the stacks become thread-local.  If multiple
    threads access them, then the state is not shared.  Each thread uses its own
    value; a thread doesn't affect other threads by mutating such a stack.
    The initial value for every thread's stack is set to the current value
    of the stack when `switch_to_thread_local()` was first called.
    """
    if not self._stack_state_is_thread_local:
      self._stack_state_is_thread_local = True
  @property
  def _device_function_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where device_function_stack doesn't yet
      # exist.
      # pylint: disable=protected-access
      if not hasattr(self._thread_local, "_device_function_stack"):
        stack_copy_for_this_thread = self._graph_device_function_stack.copy()
        self._thread_local._device_function_stack = stack_copy_for_this_thread
      return self._thread_local._device_function_stack
      # pylint: enable=protected-access
    else:
      return self._graph_device_function_stack
  @property
  def _device_functions_outer_to_inner(self):
    user_device_specs = self._device_function_stack.peek_objs()
    device_functions = [spec.function for spec in user_device_specs]
    device_functions_outer_to_inner = list(reversed(device_functions))
    return device_functions_outer_to_inner
  def _snapshot_device_function_stack_metadata(self):
    """Return device function stack as a list of TraceableObjects.
    Returns:
      [traceable_stack.TraceableObject, ...] where each TraceableObject's .obj
      member is a displayable name for the user's argument to Graph.device, and
      the filename and lineno members point to the code location where
      Graph.device was called directly or indirectly by the user.
    """
    snapshot = []
    for obj in self._device_function_stack.peek_traceable_objs():
      obj_copy = obj.copy_metadata()
      obj_copy.obj = obj.obj.display_name
      snapshot.append(obj_copy)
    return snapshot
  @_device_function_stack.setter
  def _device_function_stack(self, device_function_stack):
    if self._stack_state_is_thread_local:
      # pylint: disable=protected-access
      self._thread_local._device_function_stack = device_function_stack
      # pylint: enable=protected-access
    else:
      self._graph_device_function_stack = device_function_stack
  @property
  def _colocation_stack(self):
    """Return thread-local copy of colocation stack."""
    if self._stack_state_is_thread_local:
      # This may be called from a thread where colocation_stack doesn't yet
      # exist.
      # pylint: disable=protected-access
      if not hasattr(self._thread_local, "_colocation_stack"):
        stack_copy_for_this_thread = self._graph_colocation_stack.copy()
        self._thread_local._colocation_stack = stack_copy_for_this_thread
      return self._thread_local._colocation_stack
      # pylint: enable=protected-access
    else:
      return self._graph_colocation_stack
  def _snapshot_colocation_stack_metadata(self):
    """Return colocation stack metadata as a dictionary."""
    return {
        traceable_obj.obj.name: traceable_obj.copy_metadata()
        for traceable_obj in self._colocation_stack.peek_traceable_objs()
    }
  @_colocation_stack.setter
  def _colocation_stack(self, colocation_stack):
    if self._stack_state_is_thread_local:
      # pylint: disable=protected-access
      self._thread_local._colocation_stack = colocation_stack
      # pylint: enable=protected-access
    else:
      self._graph_colocation_stack = colocation_stack
  @property
  def _control_dependencies_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where control_dependencies_stack
      # doesn't yet exist.
      if not hasattr(self._thread_local, "_control_dependencies_stack"):
        self._thread_local._control_dependencies_stack = (
            self._graph_control_dependencies_stack[:])
      return self._thread_local._control_dependencies_stack
    else:
      return self._graph_control_dependencies_stack
  @_control_dependencies_stack.setter
  def _control_dependencies_stack(self, control_dependencies):
    if self._stack_state_is_thread_local:
      self._thread_local._control_dependencies_stack = control_dependencies
    else:
      self._graph_control_dependencies_stack = control_dependencies
  @property
  def _distribution_strategy_stack(self):
    """A stack to maintain distribution strategy context for each thread."""
    if not hasattr(self._thread_local, "_distribution_strategy_stack"):
      self._thread_local._distribution_strategy_stack = []  # pylint: disable=protected-access
    return self._thread_local._distribution_strategy_stack  # pylint: disable=protected-access
  @_distribution_strategy_stack.setter
  def _distribution_strategy_stack(self, _distribution_strategy_stack):
    self._thread_local._distribution_strategy_stack = (  # pylint: disable=protected-access
        _distribution_strategy_stack)
  @property
  def _global_distribute_strategy_scope(self):
    """For implementing `tf.distribute.set_strategy()`."""
    if not hasattr(self._thread_local, "distribute_strategy_scope"):
      self._thread_local.distribute_strategy_scope = None
    return self._thread_local.distribute_strategy_scope
  @_global_distribute_strategy_scope.setter
  def _global_distribute_strategy_scope(self, distribute_strategy_scope):
    self._thread_local.distribute_strategy_scope = (distribute_strategy_scope)
  def _mutation_lock(self):
    """Returns a lock to guard code that creates & mutates ops.
    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_MUTATION_LOCK_GROUP)
  def _session_run_lock(self):
    """Returns a lock to guard code for Session.run.
    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_SESSION_RUN_LOCK_GROUP)
