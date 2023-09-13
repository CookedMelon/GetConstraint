@tf_export("debugging.Assert", "Assert")
@dispatch.add_dispatch_support
@tf_should_use.should_use_result
def Assert(condition, data, summarize=None, name=None):
  """Asserts that the given condition is true.
  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.
  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
  Returns:
    assert_op: An `Operation` that, when executed, raises a
    `tf.errors.InvalidArgumentError` if `condition` is not true.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    @compatibility(TF1)
    When in TF V1 mode (that is, outside `tf.function`) Assert needs a control
    dependency on the output to ensure the assertion executes:
  ```python
  # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  with tf.control_dependencies([assert_op]):
    ... code using x ...
  ```
    @end_compatibility
  """
  if context.executing_eagerly():
    if not condition:
      xs = ops.convert_n_to_tensor(data)
      data_str = [_summarize_eager(x, summarize) for x in xs]
      raise errors.InvalidArgumentError(
          node_def=None,
          op=None,
          message="Expected '%s' to be true. Summarized data: %s" %
          (condition, "\n".join(data_str)))
    return
  with ops.name_scope(name, "Assert", [condition, data]) as name:
    xs = ops.convert_n_to_tensor(data)
    if all(x.dtype in {dtypes.string, dtypes.int32} for x in xs):
      # As a simple heuristic, we assume that string and int32 are
      # on host to avoid the need to use cond. If it is not case,
      # we will pay the price copying the tensor to host memory.
      return gen_logging_ops._assert(condition, data, summarize, name="Assert")  # pylint: disable=protected-access
    else:
      condition = ops.convert_to_tensor(condition, name="Condition")
      def true_assert():
        return gen_logging_ops._assert(  # pylint: disable=protected-access
            condition, data, summarize, name="Assert")
      guarded_assert = cond.cond(
          condition,
          gen_control_flow_ops.no_op,
          true_assert,
          name="AssertGuard")
      if context.executing_eagerly():
        return
      return guarded_assert.op
