@tf_export("name_scope", v1=[])
class name_scope_v2(object):
  """A context manager for use when defining a Python op.
  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.
  For example, to define a new Python op called `my_op`:
  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```
  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.
  Inside a `tf.function`, if the scope name already exists, the name will be
  made unique by appending `_n`. For example, calling `my_op` the second time
  will generate `MyOp_1/a`, etc.
  """
  __slots__ = ["_name", "_exit_fns"]
  def __init__(self, name):
    """Initialize the context manager.
    Args:
      name: The prefix to use on all names created within the name scope.
    Raises:
      ValueError: If name is not a string.
    """
    if not isinstance(name, str):
      raise ValueError("name for name_scope must be a string.")
    self._name = name
    self._exit_fns = []
  @property
  def name(self):
    return self._name
  def __enter__(self):
    """Start the scope block.
    Returns:
      The scope name.
    """
    ctx = context.context()
    if ctx.executing_eagerly():
      # Names are not auto-incremented in eager mode.
      # A trailing slash breaks out of nested name scopes, indicating a
      # fully specified scope name, for compatibility with Graph.name_scope.
      # This also prevents auto-incrementing.
      old_name = ctx.scope_name
      name = self._name
      if not name:
        scope_name = ""
      elif name[-1] == "/":
        scope_name = name
      elif old_name:
        scope_name = old_name + name + "/"
      else:
        scope_name = name + "/"
      ctx.scope_name = scope_name
      def _restore_name_scope(*_):
        ctx.scope_name = old_name
      self._exit_fns.append(_restore_name_scope)
    else:
      scope = get_default_graph().name_scope(self._name)
      scope_name = scope.__enter__()
      self._exit_fns.append(scope.__exit__)
    return scope_name
  def __exit__(self, type_arg, value_arg, traceback_arg):
    self._exit_fns.pop()(type_arg, value_arg, traceback_arg)
    return False  # False values do not suppress exceptions
  def __getstate__(self):
    return self._name, self._exit_fns
  def __setstate__(self, state):
    self._name = state[0]
    self._exit_fns = state[1]
