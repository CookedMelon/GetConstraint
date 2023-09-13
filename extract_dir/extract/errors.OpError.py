@tf_export("errors.OpError", v1=["errors.OpError", "OpError"])
@deprecation.deprecated_endpoints("OpError")
class OpError(Exception):
  """The base class for TensorFlow exceptions.
  Usually, TensorFlow will raise a more specific subclass of `OpError` from the
  `tf.errors` module.
  """
  def __init__(self, node_def, op, message, error_code, *args):
    """Creates a new `OpError` indicating that a particular op failed.
    Args:
      node_def: The `node_def_pb2.NodeDef` proto representing the op that
        failed, if known; otherwise None.
      op: The `ops.Operation` that failed, if known; otherwise None. During
        eager execution, this field is always `None`.
      message: The message string describing the failure.
      error_code: The `error_codes_pb2.Code` describing the error.
      *args: If not empty, it should contain a dictionary describing details
        about the error. This argument is inspired by Abseil payloads:
        https://github.com/abseil/abseil-cpp/blob/master/absl/status/status.h
    """
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code
    if args:
      self._experimental_payloads = args[0]
    else:
      self._experimental_payloads = {}
  def __reduce__(self):
    # Allow the subclasses to accept less arguments in their __init__.
    init_argspec = tf_inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args
  @property
  def message(self):
    """The error message that describes the error."""
    return self._message
  @property
  def op(self):
    """The operation that failed, if known.
    *N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
    or `Recv` op, there will be no corresponding
    `tf.Operation`
    object.  In that case, this will return `None`, and you should
    instead use the `tf.errors.OpError.node_def` to
    discover information about the op.
    Returns:
      The `Operation` that failed, or None.
    """
    return self._op
  @property
  def error_code(self):
    """The integer error code that describes the error."""
    return self._error_code
  @property
  def node_def(self):
    """The `NodeDef` proto representing the op that failed."""
    return self._node_def
  @property
  def experimental_payloads(self):
    """A dictionary describing the details of the error."""
    return self._experimental_payloads
  def __str__(self):
    if self._op is not None:
      output = [
          "%s\n\nOriginal stack trace for %r:\n" % (
              self.message,
              self._op.name,
          )
      ]
      curr_traceback_list = traceback.format_list(self._op.traceback or [])
      output.extend(curr_traceback_list)
      # pylint: disable=protected-access
      original_op = self._op._original_op
      # pylint: enable=protected-access
      while original_op is not None:
        output.append(
            "\n...which was originally created as op %r, defined at:\n" %
            (original_op.name,))
        prev_traceback_list = curr_traceback_list
        curr_traceback_list = traceback.format_list(original_op.traceback or [])
        # Attempt to elide large common subsequences of the subsequent
        # stack traces.
        #
        # TODO(mrry): Consider computing the actual longest common subsequence.
        is_eliding = False
        elide_count = 0
        last_elided_line = None
        for line, line_in_prev in zip(curr_traceback_list, prev_traceback_list):
          if line == line_in_prev:
            if is_eliding:
              elide_count += 1
              last_elided_line = line
            else:
              output.append(line)
              is_eliding = True
              elide_count = 0
          else:
            if is_eliding:
              if elide_count > 0:
                output.extend([
                    "[elided %d identical lines from previous traceback]\n" %
                    (elide_count - 1,), last_elided_line
                ])
              is_eliding = False
            output.extend(line)
        # pylint: disable=protected-access
        original_op = original_op._original_op
        # pylint: enable=protected-access
      return "".join(output)
    else:
      return self.message
