@tf_export("Operation")
class Operation(pywrap_tf_session.PyOperation):
  """Represents a graph node that performs computation on tensors.
  An `Operation` is a node in a `tf.Graph` that takes zero or more `Tensor`
  objects as input, and produces zero or more `Tensor` objects as output.
  Objects of type `Operation` are created by calling a Python op constructor
  (such as `tf.matmul`) within a `tf.function` or under a `tf.Graph.as_default`
  context manager.
  For example, within a `tf.function`, `c = tf.matmul(a, b)` creates an
  `Operation` of type "MatMul" that takes tensors `a` and `b` as input, and
  produces `c` as output.
  If a `tf.compat.v1.Session` is used, an `Operation` of a `tf.Graph` can be
  executed by passing it to `tf.Session.run`. `op.run()` is a shortcut for
  calling `tf.compat.v1.get_default_session().run(op)`.
  """
  @classmethod
  def from_node_def(
      cls,
      node_def,
      g,
      inputs=None,
      output_types=None,
      control_inputs=None,
      input_types=None,
      original_op=None,
      op_def=None,
  ):
    r"""Creates an `Operation`.
    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:
        [A-Za-z0-9.][A-Za-z0-9_.\\-/]*
    Args:
      node_def: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`. Used for
        attributes of `node_def_pb2.NodeDef`, typically `name`, `op`, and
        `device`.  The `input` attribute is irrelevant here as it will be
        computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the `Tensors`
        computed by this operation.  The length of this list indicates the
        number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a control
        dependency.
      input_types: List of `DType` objects representing the types of the tensors
        accepted by the `Operation`.  By default uses `[x.dtype.base_dtype for x
        in inputs]`.  Operations that expect reference-typed inputs must specify
        these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the op type
        that this `Operation` represents.
    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.
    Returns:
      Operation object.
    """
    if not isinstance(g, Graph):
      raise TypeError(f"Argument g must be a Graph. "
                      f"Received an instance of type {type(g)}")
    if not isinstance(node_def, node_def_pb2.NodeDef):
      raise TypeError(f"Argument node_def must be a NodeDef. "
                      f"Received an instance of type: {type(node_def)}.")
    if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
      raise ValueError(
          f"Cannot create a tensor proto whose content is larger than 2GB. "
          f"Size of tensor is {node_def.ByteSize()} bytes.")
    # TODO(mdan): This does not belong here. Graph::AddNode should handle it.
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
      raise ValueError(
          f"`{node_def.name}` is not a valid node name. "
          f"Accepted names conform to Regex /{_VALID_OP_NAME_REGEX}/")
    # FIXME(b/225400189): output_types is unused. Consider remove it from
    # the argument list.
    del output_types
    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError(f"Argument inputs shall be a list of Tensors. "
                      f"Received an instance of type {type(inputs)}")
    for a in inputs:
      if not isinstance(a, Tensor):
        raise TypeError(f"Items of argument inputs shall be Tensor. "
                        f"Received an instance of type {type(a)}.")
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]
    else:
      if not all(
          x.is_compatible_with(i.dtype) for i, x in zip(inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" %
                        (node_def.name, [i.dtype for i in inputs], input_types))
    # Build the list of control inputs.
    control_input_ops = []
    if control_inputs:
      for c in control_inputs:
        control_op = None
        if isinstance(c, Operation):
          control_op = c
        elif isinstance(c, (Tensor, internal.IndexedSlices)):
          control_op = c.op
        else:
          raise TypeError(f"Control input must be an Operation, "
                          f"a Tensor, or IndexedSlices. "
                          f"Received an instance of type {type(c)}.")
        control_input_ops.append(control_op)
    # Initialize c_op from node_def and other inputs
    c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
    self = Operation(c_op, GraphTensor)
    self._init(g)
    self._original_op = original_op
    # Post process for control flows.
    self._control_flow_post_processing(input_tensors=inputs)
    # Removes this frame from the Python traceback.
    # We adjust stacklevel directly to avoid triggering serialization.
    if self.traceback is not None:
      self.traceback._stacklevel += 1  # pylint: disable=protected-access
    return self
  @classmethod
  def _from_c_op(cls, c_op, g):
    """Create an Operation from a TF_Operation.
    For internal use only: This is useful for creating Operation for ops
    indirectly created by C API methods, e.g. the ops created by
    TF_ImportGraphDef.
    Args:
      c_op: a TF_Operation.
      g: A Graph.
    Returns:
      an Operation object.
    """
    self = Operation(c_op, GraphTensor)
    self._init(g)
    return self
  def _init(self, graph):
    """Initializes Operation from a TF_Operation."""
    self.graph = graph
    self._original_op = None
    # This will be set by self.inputs.
    self._inputs_val = None
    # List of _UserDevSpecs holding code location of device context manager
    # invocations and the users original argument to them.
    self._device_code_locations = None
    # Dict mapping op name to file and line information for op colocation
    # context managers.
    self._colocation_code_locations = None
    self._control_flow_context = self.graph._get_control_flow_context()  # pylint: disable=protected-access
    # Gradient function for this op. There are three ways to specify gradient
    # function, and first available gradient gets used, in the following order.
    # 1. self._gradient_function
    # 2. Gradient name registered by "_gradient_op_type" attribute.
    # 3. Gradient name registered by op.type.
    self._gradient_function = None
    self._init_outputs()
    self._id_value = self.graph._add_op(self)  # pylint: disable=protected-access
  def _control_flow_post_processing(self, input_tensors=None):
    """Add this op to its control flow context.
    This may add new ops and change this op's inputs. self.inputs must be
    available before calling this method.
    Args:
      input_tensors: (Optional.) A list of `Tensors` corresponding to the inputs
        of this op, which should be equivalent to `self.inputs`. Pass this
        argument to avoid evaluating `self.inputs` unnecessarily.
    """
    if input_tensors is None:
      input_tensors = self.inputs
    for input_tensor in input_tensors:
      control_flow_util.CheckInputFromValidContext(self, input_tensor.op)
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)
  def colocation_groups(self):
    """Returns the list of colocation groups of the op."""
    default_colocation_group = [compat.as_bytes("loc:@%s" % self.name)]
    try:
      class_attr = self.get_attr("_class")
    except ValueError:
      # This op has no explicit colocation group, so it is itself its
      # own root of a colocation group.
      return default_colocation_group
    attr_groups = [
        class_name for class_name in class_attr
        if class_name.startswith(b"loc:@")
    ]
    # If there are no colocation groups in the explicit _class field,
    # return the default colocation group.
    return attr_groups if attr_groups else default_colocation_group
  def values(self):
    """DEPRECATED: Use outputs."""
    return tuple(self.outputs)
  def _get_control_flow_context(self):
    """Returns the control flow context of this op.
    Returns:
      A context object.
    """
    return self._control_flow_context
  def _set_control_flow_context(self, ctx):
    """Sets the current control flow context of this op.
    Args:
      ctx: a context object.
    """
    self._control_flow_context = ctx
  @property
  def _id(self):
    """The unique integer id of this operation."""
    return self._id_value
  @property
  def device(self):
    """The name of the device to which this op has been assigned, if any.
    Returns:
      The string name of the device to which this op has been
      assigned, or an empty string if it has not been assigned to a
      device.
    """
    return pywrap_tf_session.TF_OperationDevice(self._c_op)
  @property
  def _device_assignments(self):
    """Code locations for device context managers active at op creation.
    This property will return a list of traceable_stack.TraceableObject
    instances where .obj is a string representing the assigned device
    (or information about the function that would be applied to this op
    to compute the desired device) and the filename and lineno members
    record the location of the relevant device context manager.
    For example, suppose file_a contained these lines:
      file_a.py:
        15: with tf.device('/gpu:0'):
        16:   node_b = tf.constant(4, name='NODE_B')
    Then a TraceableObject t_obj representing the device context manager
    would have these member values:
      t_obj.obj -> '/gpu:0'
      t_obj.filename = 'file_a.py'
      t_obj.lineno = 15
    and node_b.op._device_assignments would return the list [t_obj].
    Returns:
      [str: traceable_stack.TraceableObject, ...] as per this method's
      description, above.
    """
    return self._device_code_locations or []
  @property
  def _colocation_dict(self):
    """Code locations for colocation context managers active at op creation.
    This property will return a dictionary for which the keys are nodes with
    which this Operation is colocated, and for which the values are
    traceable_stack.TraceableObject instances.  The TraceableObject instances
    record the location of the relevant colocation context manager but have the
    "obj" field set to None to prevent leaking private data.
    For example, suppose file_a contained these lines:
      file_a.py:
        14: node_a = tf.constant(3, name='NODE_A')
        15: with tf.compat.v1.colocate_with(node_a):
        16:   node_b = tf.constant(4, name='NODE_B')
    Then a TraceableObject t_obj representing the colocation context manager
    would have these member values:
      t_obj.obj -> None
      t_obj.filename = 'file_a.py'
      t_obj.lineno = 15
    and node_b.op._colocation_dict would return the dictionary
      { 'NODE_A': t_obj }
    Returns:
      {str: traceable_stack.TraceableObject} as per this method's description,
      above.
    """
    locations_dict = self._colocation_code_locations or {}
    return locations_dict.copy()
  @property
  def _output_types(self):
    """List this operation's output types.
    Returns:
      List of the types of the Tensors computed by this operation.
      Each element in the list is an integer whose value is one of
      the TF_DataType enums defined in pywrap_tf_session.h
      The length of this list indicates the number of output endpoints
      of the operation.
    """
    num_outputs = pywrap_tf_session.TF_OperationNumOutputs(self._c_op)
    output_types = [
        int(pywrap_tf_session.TF_OperationOutputType(self._tf_output(i)))
        for i in range(num_outputs)
    ]
    return output_types
  def _set_device(self, device):  # pylint: disable=redefined-outer-name
    """Set the device of this operation.
    Args:
      device: string or device..  The device to set.
    """
    self._set_device_from_string(compat.as_str(_device_string(device)))
  def _update_input(self, index, tensor):
    """Update the input to this operation at the given index.
    NOTE: This is for TF internal use only. Please don't use it.
    Args:
      index: the index of the input to update.
      tensor: the Tensor to be used as the input at the given index.
    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)
    # Reset cached inputs.
    self._inputs_val = None
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      pywrap_tf_session.UpdateEdge(
          c_graph,
          tensor._as_tf_output(),  # pylint: disable=protected-access
          self._tf_input(index))
  def _add_while_inputs(self, tensors):
    """See AddWhileInputHack in python_api.h.
    NOTE: This is for TF internal use only. Please don't use it.
    Args:
      tensors: list of Tensors
    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      for tensor in tensors:
        if not isinstance(tensor, Tensor):
          raise TypeError("tensor must be a Tensor: %s" % tensor)
        _assert_same_graph(self, tensor)
        # Reset cached inputs.
        self._inputs_val = None
        pywrap_tf_session.AddWhileInputHack(
            c_graph,  # pylint: disable=protected-access
            tensor._as_tf_output(),  # pylint: disable=protected-access
            self._c_op)
  def __str__(self):
    return str(self.node_def)
  def __repr__(self):
    return "<tf.Operation '%s' type=%s>" % (self.name, self.type)
  def __tf_tensor__(self, dtype=None, name=None):
    """Raises a helpful error."""
    raise TypeError("can't convert Operation '{}' to Tensor".format(self.name))
  @property
  def inputs(self):
    """The sequence of `Tensor` objects representing the data inputs of this op."""
    if self._inputs_val is None:
      # pylint: disable=protected-access
      self._inputs_val = tuple(
          self.graph._get_tensor_by_tf_output(i)
          for i in pywrap_tf_session.GetOperationInputs(self._c_op))
      # pylint: enable=protected-access
    return self._inputs_val
  @property
  def _input_types(self):
    num_inputs = pywrap_tf_session.TF_OperationNumInputs(self._c_op)
    input_types = [
        dtypes.as_dtype(
            pywrap_tf_session.TF_OperationInputType(self._tf_input(i)))
        for i in range(num_inputs)
    ]
    return input_types
  @property
  def traceback(self):
    """Returns the call stack from when this operation was constructed."""
    # FIXME(b/225423591): This object contains a dangling reference if _c_op
    # goes out of scope.
    return pywrap_tf_session.TF_OperationGetStackTrace(self._c_op)
  @property
  def node_def(self):
    return node_def_pb2.NodeDef.FromString(self._node_def)
  @property
  def op_def(self):
    return op_def_pb2.OpDef.FromString(self._op_def)
  def _set_attr(self, attr_name, attr_value):
    """Private method used to set an attribute in the node_def."""
    buf = pywrap_tf_session.TF_NewBufferFromString(
        compat.as_bytes(attr_value.SerializeToString()))
    try:
      self._set_attr_with_buf(attr_name, buf)
    finally:
      pywrap_tf_session.TF_DeleteBuffer(buf)
  def _set_attr_with_buf(self, attr_name, attr_buf):
    """Set an attr in the node_def with a pre-allocated buffer."""
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      # pylint: disable=protected-access
      pywrap_tf_session.SetAttr(c_graph, self._c_op, attr_name, attr_buf)
      # pylint: enable=protected-access
  def _set_func_attr(self, attr_name, func_name):
    """Private method used to set a function attribute in the node_def."""
    func = attr_value_pb2.NameAttrList(name=func_name)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(func=func))
  def _set_func_list_attr(self, attr_name, func_names):
    """Private method used to set a list(function) attribute in the node_def."""
    funcs = [attr_value_pb2.NameAttrList(name=func_name)
             for func_name in func_names]
    funcs_list = attr_value_pb2.AttrValue.ListValue(func=funcs)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=funcs_list))
  def _set_type_list_attr(self, attr_name, types):
    """Private method used to set a list(type) attribute in the node_def."""
    if not types:
      return
    if isinstance(types[0], dtypes.DType):
      types = [dt.as_datatype_enum for dt in types]
    types_list = attr_value_pb2.AttrValue.ListValue(type=types)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=types_list))
  def _set_shape_list_attr(self, attr_name, shapes):
    """Private method used to set a list(shape) attribute in the node_def."""
    shapes = [s.as_proto() for s in shapes]
    shapes_list = attr_value_pb2.AttrValue.ListValue(shape=shapes)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=shapes_list))
  def _clear_attr(self, attr_name):
    """Private method used to clear an attribute in the node_def."""
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      # pylint: disable=protected-access
      pywrap_tf_session.ClearAttr(c_graph, self._c_op, attr_name)
      # pylint: enable=protected-access
  def get_attr(self, name):
    """Returns the value of the attr of this op with the given `name`.
    Args:
      name: The name of the attr to fetch.
    Returns:
      The value of the attr, as a Python object.
    Raises:
      ValueError: If this op does not have an attr with the given `name`.
    """
    fields = ("s", "i", "f", "b", "type", "shape", "tensor", "func")
    try:
      with c_api_util.tf_buffer() as buf:
        pywrap_tf_session.TF_OperationGetAttrValueProto(self._c_op, name, buf)
        data = pywrap_tf_session.TF_GetBuffer(buf)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)
    x = attr_value_pb2.AttrValue()
    x.ParseFromString(data)
    oneof_value = x.WhichOneof("value")
    if oneof_value is None:
      return []
    if oneof_value == "list":
      for f in fields:
        if getattr(x.list, f):
          if f == "type":
            return [dtypes.as_dtype(t) for t in x.list.type]
          else:
            return list(getattr(x.list, f))
      return []
    if oneof_value == "type":
      return dtypes.as_dtype(x.type)
    assert oneof_value in fields, "Unsupported field type in " + str(x)
    return getattr(x, oneof_value)
  def _get_attr_type(self, name):
    """Returns the `DType` value of the attr of this op with the given `name`."""
    try:
      dtype_enum = pywrap_tf_session.TF_OperationGetAttrType(self._c_op, name)
      return _DTYPES_INTERN_TABLE[dtype_enum]
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)
  def _get_attr_bool(self, name):
    """Returns the `bool` value of the attr of this op with the given `name`."""
    try:
      return pywrap_tf_session.TF_OperationGetAttrBool(self._c_op, name)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)
  def _get_attr_int(self, name):
    """Returns the `int` value of the attr of this op with the given `name`."""
    try:
      return pywrap_tf_session.TF_OperationGetAttrInt(self._c_op, name)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)
  def experimental_set_type(self, type_proto):
    """Sets the corresponding node's `experimental_type` field.
    See the description of `NodeDef.experimental_type` for more info.
    Args:
      type_proto: A FullTypeDef proto message. The root type_if of this object
        must be `TFT_PRODUCT`, even for ops which only have a singlre return
        value.
    """
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      if (type_proto.type_id
          not in (full_type_pb2.TFT_UNSET, full_type_pb2.TFT_PRODUCT)):
        raise ValueError("error setting the type of ", self.name,
                         ": expected TFT_UNSET or TFT_PRODUCT, got ",
                         type_proto.type_id)
      pywrap_tf_session.SetFullType(c_graph, self._c_op,
                                    type_proto.SerializeToString())  # pylint:disable=protected-access
  def run(self, feed_dict=None, session=None):
    """Runs this operation in a `Session`.
    Calling this method will execute all preceding operations that
    produce the inputs needed for this operation.
    *N.B.* Before invoking `Operation.run()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.
    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to run to this operation. If
        none, the default session will be used.
    """
    _run_using_default_session(self, feed_dict, self.graph, session)
