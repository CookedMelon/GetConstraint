@tf_export("control_dependencies")
def control_dependencies(control_inputs):
  """Wrapper for `Graph.control_dependencies()` using the default graph.
  See `tf.Graph.control_dependencies` for more details.
  In TensorFlow 2 with eager and/or Autograph, you should not need this method
  most of the times, as ops execute in the expected order thanks to automatic
  control dependencies. Only use it to manually control ordering, for example as
  a workaround to known issues such as `tf.function` with `tf.debugging.assert*`
  and `tf.py_function`.
  For example:
  >>> @tf.function(
  ...   input_signature=[tf.TensorSpec([None, None], tf.float32),
  ...                    tf.TensorSpec([None, None], tf.float32)])
  ... def my_assert_func_1(x, bias):
  ...   # `tf.function` attempts to execute `tf.math.add` in parallel to
  ...   # `assert_equal`. As a result an error can get raised from `tf.math.add`
  ...   # without triggering the assertion error.
  ...   tf.assert_equal(tf.shape(x)[1],
  ...                   tf.shape(bias)[1],
  ...                   message='bad shape')
  ...   return x + bias
  >>> # Error raised in either `add` or `assert`
  >>> my_assert_func_1(tf.ones((2, 5)), tf.ones((2, 7)))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...
  >>> @tf.function(
  ...   input_signature=[tf.TensorSpec([None, None], tf.float32),
  ...                    tf.TensorSpec([None, None], tf.float32)])
  ... def my_assert_func_2(x, bias):
  ...   with tf.control_dependencies(
  ...       [tf.assert_equal(tf.shape(x)[1],
  ...                       tf.shape(bias)[1],
  ...                       message='bad shape')]):
  ...     return x + bias
  >>> # Error raised in `assert`
  >>> my_assert_func_2(tf.ones((2, 5)), tf.ones((2, 7)))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...
  When eager execution is enabled, any callable object in the `control_inputs`
  list will be called.
  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which must be
      executed or computed before running the operations defined in the context.
      Can also be `None` to clear the control dependencies. If eager execution
      is enabled, any callable object in the `control_inputs` list will be
      called.
  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  """
  if context.executing_eagerly():
    if control_inputs:
      # Execute any pending callables.
      for control in control_inputs:
        if callable(control):
          control()
    return NullContextmanager()
  else:
    return get_default_graph().control_dependencies(control_inputs)
