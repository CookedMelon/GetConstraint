@tf_export("init_scope")
@tf_contextlib.contextmanager
def init_scope():
  """A context manager that lifts ops out of control-flow scopes and function-building graphs.
  There is often a need to lift variable initialization ops out of control-flow
  scopes, function-building graphs, and gradient tapes. Entering an
  `init_scope` is a mechanism for satisfying these desiderata. In particular,
  entering an `init_scope` has three effects:
    (1) All control dependencies are cleared the moment the scope is entered;
        this is equivalent to entering the context manager returned from
        `control_dependencies(None)`, which has the side-effect of exiting
        control-flow scopes like `tf.cond` and `tf.while_loop`.
    (2) All operations that are created while the scope is active are lifted
        into the lowest context on the `context_stack` that is not building a
        graph function. Here, a context is defined as either a graph or an eager
        context. Every context switch, i.e., every installation of a graph as
        the default graph and every switch into eager mode, is logged in a
        thread-local stack called `context_switches`; the log entry for a
        context switch is popped from the stack when the context is exited.
        Entering an `init_scope` is equivalent to crawling up
        `context_switches`, finding the first context that is not building a
        graph function, and entering it. A caveat is that if graph mode is
        enabled but the default graph stack is empty, then entering an
        `init_scope` will simply install a fresh graph as the default one.
    (3) The gradient tape is paused while the scope is active.
  When eager execution is enabled, code inside an init_scope block runs with
  eager execution enabled even when tracing a `tf.function`. For example:
  ```python
  tf.compat.v1.enable_eager_execution()
  @tf.function
  def func():
    # A function constructs TensorFlow graphs,
    # it does not execute eagerly.
    assert not tf.executing_eagerly()
    with tf.init_scope():
      # Initialization runs with eager execution enabled
      assert tf.executing_eagerly()
  ```
  Raises:
    RuntimeError: if graph state is incompatible with this initialization.
  """
  # pylint: enable=g-doc-return-or-yield,line-too-long
  if context.executing_eagerly():
    # Fastpath.
    with record.stop_recording():
      yield
  else:
    # Retrieve the active name scope: entering an `init_scope` preserves
    # the name scope of the current context.
    scope = get_default_graph().get_name_scope()
    if scope and scope[-1] != "/":
      # Names that end with trailing slashes are treated by `name_scope` as
      # absolute.
      scope = scope + "/"
    outer_context, innermost_nonempty_device_stack = (
        _get_outer_context_and_inner_device_stack())
    outer_graph = None
    outer_device_stack = None
    try:
      with outer_context(), name_scope(
          scope, skip_on_eager=False), control_dependencies(
              None), record.stop_recording():
        context_manager = NullContextmanager
        context_manager_input = None
        if not context.executing_eagerly():
          # The device stack is preserved when lifting into a graph. Eager
          # execution doesn't implement device stacks and in particular it
          # doesn't support device functions, so in general it's not possible
          # to do the same when lifting into the eager context.
          outer_graph = get_default_graph()
          outer_device_stack = outer_graph._device_function_stack  # pylint: disable=protected-access
          outer_graph._device_function_stack = innermost_nonempty_device_stack  # pylint: disable=protected-access
        elif innermost_nonempty_device_stack is not None:
          for device_spec in innermost_nonempty_device_stack.peek_objs():
            if device_spec.function is None:
              break
            if device_spec.raw_string:
              context_manager = context.device
              context_manager_input = device_spec.raw_string
              break
            # It is currently not possible to have a device function in V2,
            # but in V1 we are unable to apply device functions in eager mode.
            # This means that we will silently skip some of the entries on the
            # device stack in V1 + eager mode.
        with context_manager(context_manager_input):
          yield
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # try-block (just above).
      if outer_graph is not None:
        outer_graph._device_function_stack = outer_device_stack  # pylint: disable=protected-access
