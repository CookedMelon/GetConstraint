@tf_export("CriticalSection")
class CriticalSection:
  """Critical section.
  A `CriticalSection` object is a resource in the graph which executes subgraphs
  in **serial** order.  A common example of a subgraph one may wish to run
  exclusively is the one given by the following function:
  ```python
  v = resource_variable_ops.ResourceVariable(0.0, name="v")
  def count():
    value = v.read_value()
    with tf.control_dependencies([value]):
      with tf.control_dependencies([v.assign_add(1)]):
        return tf.identity(value)
  ```
  Here, a snapshot of `v` is captured in `value`; and then `v` is updated.
  The snapshot value is returned.
  If multiple workers or threads all execute `count` in parallel, there is no
  guarantee that access to the variable `v` is atomic at any point within
  any thread's calculation of `count`.  In fact, even implementing an atomic
  counter that guarantees that the user will see each value `0, 1, ...,` is
  currently impossible.
  The solution is to ensure any access to the underlying resource `v` is
  only processed through a critical section:
  ```python
  cs = CriticalSection()
  f1 = cs.execute(count)
  f2 = cs.execute(count)
  output = f1 + f2
  session.run(output)
  ```
  The functions `f1` and `f2` will be executed serially, and updates to `v`
  will be atomic.
  **NOTES**
  All resource objects, including the critical section and any captured
  variables of functions executed on that critical section, will be
  colocated to the same device (host and cpu/gpu).
  When using multiple critical sections on the same resources, there is no
  guarantee of exclusive access to those resources.  This behavior is disallowed
  by default (but see the kwarg `exclusive_resource_access`).
  For example, running the same function in two separate critical sections
  will not ensure serial execution:
  ```python
  v = tf.compat.v1.get_variable("v", initializer=0.0, use_resource=True)
  def accumulate(up):
    x = v.read_value()
    with tf.control_dependencies([x]):
      with tf.control_dependencies([v.assign_add(up)]):
        return tf.identity(x)
  ex1 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  ex2 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  bad_sum = ex1 + ex2
  sess.run(v.initializer)
  sess.run(bad_sum)  # May return 0.0
  ```
  """
  def __init__(self, name=None, shared_name=None,
               critical_section_def=None, import_scope=None):
    """Creates a critical section."""
    context.ensure_initialized()
    if critical_section_def and name is not None:
      raise ValueError(f"Arguments critical_section_def={critical_section_def} "
                       f"and shared_name={shared_name} are mutually exclusive. "
                       "Please only specify one of them.")
    if critical_section_def:
      raise ValueError("Argument `critical_section_def` is not supported.")
    else:
      self._init_from_args(name, shared_name)
  def _init_from_args(self, name, shared_name):  # pylint: disable=invalid-name
    """Initialize the CriticalSection from constructor arguments."""
    with ops.name_scope(name, "CriticalSection", []) as name:
      with ops.init_scope():
        # pylint: disable=protected-access
        container = ops.get_default_graph()._container
        # pylint: enable=protected-access
        if shared_name is None:
          shared_name = name
        if container is None:
          container = ""
        self._handle = gen_resource_variable_ops.mutex_v2(
            shared_name=shared_name, container=container, name=name)
        # Get a uniquely identifying signature for the handle.
        self._signature = (
            container,
            # If shared_name is empty, a unique CriticalSection is created.
            shared_name or id(self._handle),
            _get_device_or_colocation(self._handle))
    if not context.executing_eagerly():
      ops.add_to_collections(CRITICAL_SECTIONS, self)
  @property
  def name(self):
    return self._handle.op.name
  def execute(self, fn, exclusive_resource_access=True, name=None):
    """Execute function `fn()` inside the critical section.
    `fn` should not accept any arguments.  To add extra arguments to when
    calling `fn` in the critical section, create a lambda:
    ```python
    critical_section.execute(lambda: fn(*my_args, **my_kwargs))
    ```
    Args:
      fn: The function to execute.  Must return at least one tensor.
      exclusive_resource_access: Whether the resources required by
        `fn` should be exclusive to this `CriticalSection`.  Default: `True`.
        You may want to set this to `False` if you will be accessing a
        resource in read-only mode in two different CriticalSections.
      name: The name to use when creating the execute operation.
    Returns:
      The tensors returned from `fn()`.
    Raises:
      ValueError: If `fn` attempts to lock this `CriticalSection` in any nested
        or lazy way that may cause a deadlock.
      ValueError: If `exclusive_resource_access == True` and
        another `CriticalSection` has an execution requesting the same
        resources as `fn``.  Note, even if `exclusive_resource_access` is
        `True`, if another execution in another `CriticalSection` was created
        without `exclusive_resource_access=True`, a `ValueError` will be raised.
    """
    with ops.name_scope(name, "critical_section_execute", []):
      # Ensure that mutex locking only happens *after* all args and
      # kwargs have been executed.  This avoids certain types of deadlocks.
      with _push_critical_section_stack(self._signature):
        lock = gen_resource_variable_ops.mutex_lock(self._handle)
        if not context.executing_eagerly():
          # NOTE(ebrevdo): This is to ensure we don't pick up spurious
          # Operations created by other threads.
          with ops.get_default_graph()._lock:  # pylint: disable=protected-access
            existing_ops = ops.get_default_graph().get_operations()
            with ops.control_dependencies([lock]):
              r = fn()
            # TODO(ebrevdo): If creating critical sections in a python loop,
            # this makes graph creation time quadratic.  Revisit if this
            # becomes a problem.
            created_ops = (set(ops.get_default_graph().get_operations())
                           .difference(existing_ops))
        else:
          with ops.control_dependencies([lock]):
            r = fn()
      if not context.executing_eagerly():
        self._add_control_dependencies_to_lock(created_ops, lock.op)
        # captured_resources is a list of resources that are directly
        # accessed only by ops created during fn(), not by any
        # ancestors of those ops in the graph.
        captured_resources = object_identity.ObjectIdentitySet([
            input_ for op in created_ops
            for input_ in op.inputs
            if input_.dtype == dtypes.resource
        ])
        # NOTE(ebrevdo): The only time self._is_self_handle() is True
        # in this call is if one of the recently created ops, within
        # the execute(), themselves attempt to access the
        # CriticalSection.  This will cause a deadlock.
        if any(self._is_self_handle(x) for x in captured_resources):
          raise ValueError(
              "Attempting to lock a CriticalSection in which we are "
              f"already running (signature={self._signature}). This is illegal "
              "and may cause deadlocks.")
        self._check_multiple_access_to_resources(
            captured_resources, exclusive_resource_access)
      r_flat = [_identity(x) for x in nest.flatten(r)]
      with ops.control_dependencies(r_flat):
        # The identity must run on the same machine as self._handle
        with ops.colocate_with(self._handle):
          # Do not use array_ops.identity as there are special
          # optimizations within TensorFlow which seem to elide it
          # even when optimizations are disabled(!).
          ensure_lock_exists = gen_resource_variable_ops.consume_mutex_lock(
              lock)
        # Make sure that if any element of r is accessed, all of
        # them are executed together.
        r = nest.pack_sequence_as(r, control_flow_ops.tuple(nest.flatten(r)))
      with ops.control_dependencies([ensure_lock_exists]):
        outputs = nest.map_structure(_identity, r)
      if not context.executing_eagerly():
        signature = _ExecutionSignature(
            op=lock.op,
            handle=self._handle,
            resources=list(captured_resources),
            exclusive_resource_access=exclusive_resource_access)
        ops.add_to_collections(
            CRITICAL_SECTION_EXECUTIONS, signature)
      return outputs
  def _add_control_dependencies_to_lock(self, created_ops, lock_op):
    """To avoid deadlocks, all args must be executed before lock_op."""
    # Get all arguments (explicit and captured) of all ops created by fn().
    all_args = set([input_.op for op in created_ops for input_ in op.inputs])
    all_args.update(
        input_op for op in created_ops for input_op in op.control_inputs)
    # Unfortunately, we can't use sets throughout because TF seems to
    # create new Operation objects for the same op sometimes; and we
    # can't rely on id(op).
    # pylint: disable=protected-access
    all_args_dict = dict((op._id, op) for op in all_args)
    # Remove ops created within fn, or that lock_op already has a
    # control dependency on.  Also remove a possible self-loop.
    for op in created_ops:
      all_args_dict.pop(op._id, None)
    for op in lock_op.control_inputs:
      all_args_dict.pop(op._id, None)
    for input_ in lock_op.inputs:
      all_args_dict.pop(input_.op._id, None)
    all_args_dict.pop(lock_op._id, None)
    all_args = all_args_dict.values()
    if not all_args:
      # No control dependencies to add; return early.
      return
    # This group is important: it ensures that any ops in all_args
    # outside the control context of the lock_op (and this fn, which
    # runs in the same context) are added to this context before
    # being added to the control dependencies of lock_op.
    all_args = control_flow_ops.group(*all_args)
    lock_op._add_control_input(all_args)
    # pylint: enable=protected-access
  def _is_self_handle(self, x):
    """Check if the tensor `x` is the same Mutex as `self._handle`."""
    if isinstance(x, ops.EagerTensor):
      return x is self._handle
    return (x.op.type == "MutexV2"
            # blank shared_name means the op will create a unique one.
            and x.op.get_attr("shared_name")
            and (x.op.get_attr("shared_name") ==
                 self._handle.op.get_attr("shared_name"))
            and (x.op.device == self._handle.op.device
                 or _get_colocation(x.op) == _get_colocation(self._handle.op)))
  def _check_multiple_access_to_resources(
      self, captured_resources, exclusive_resource_access):
    """Raise if captured_resources are accessed by another CriticalSection.
    Args:
      captured_resources: Set of tensors of type resource.
      exclusive_resource_access: Whether this execution requires exclusive
        resource access.
    Raises:
      ValueError: If any tensors in `captured_resources` are also accessed
        by another `CriticalSection`, and at least one of them requires
        exclusive resource access.
    """
    # Collections and op introspection does not work in eager
    # mode.  This is generally ok; since eager mode (as of
    # writing) executes sequentially anyway.
    for sg in ops.get_collection(CRITICAL_SECTION_EXECUTIONS):
      if self._is_self_handle(sg.handle):
        # Other executions in the same critical section are allowed.
        continue
      if not (exclusive_resource_access or sg.exclusive_resource_access):
        # Neither execution requested exclusive access.
        continue
      resource_intersection = captured_resources.intersection(sg.resources)
      if resource_intersection:
        raise ValueError(
            "This execution would access resources: "
            f"{list(resource_intersection)}. Either this lock "
            f"(CriticalSection: {self._handle}) or lock '{sg}' "
            f"(CriticalSection: {sg.handle}) requested exclusive resource "
            "access of this resource. Did you mean to call execute with "
            "keyword argument exclusive_resource_access=False?")
