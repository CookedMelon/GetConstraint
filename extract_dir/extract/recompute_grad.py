@tf_export("recompute_grad")
def recompute_grad(f):
  """Defines a function as a recompute-checkpoint for the tape auto-diff.
  Tape checkpointing is a technique to reduce the memory consumption of the
  auto-diff tape:
  - Without tape checkpointing operations and intermediate values are
  recorded to the tape for use in the backward pass.
  - With tape checkpointing, only the function call and its inputs are
  recorded. During back-propagation the `recompute_grad` custom gradient
  (`tf.custom_gradient`) recomputes the function under a localized Tape object.
  This recomputation of the function during backpropagation performs redundant
  calculation, but reduces the overall memory usage of the Tape.
  >>> y = tf.Variable(1.0)
  >>> def my_function(x):
  ...   tf.print('running')
  ...   z = x*y
  ...   return z
  >>> my_function_recompute = tf.recompute_grad(my_function)
  >>> with tf.GradientTape() as tape:
  ...   r = tf.constant(1.0)
  ...   for i in range(4):
  ...     r = my_function_recompute(r)
  running
  running
  running
  running
  >>> grad = tape.gradient(r, [y])
  running
  running
  running
  running
  Without `recompute_grad`, the tape contains all intermitate steps, and no
  recomputation is performed.
  >>> with tf.GradientTape() as tape:
  ...   r = tf.constant(1.0)
  ...   for i in range(4):
  ...     r = my_function(r)
  running
  running
  running
  running
  >>> grad = tape.gradient(r, [y])
  If `f` was a `tf.keras` `Model` or `Layer` object, methods and attributes
  such as `f.variables` are not available on the returned function `g`.
  Either keep a reference of `f` , or use `g.__wrapped__` for accessing
  these variables and methods.
  >>> def print_running_and_return(x):
  ...   tf.print("running")
  ...   return x
  >>> model = tf.keras.Sequential([
  ...   tf.keras.layers.Lambda(print_running_and_return),
  ...   tf.keras.layers.Dense(2)
  ... ])
  >>> model_recompute = tf.recompute_grad(model)
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   r = tf.constant([[1,2]])
  ...   for i in range(4):
  ...     r = model_recompute(r)
  running
  running
  running
  running
  >>> grad = tape.gradient(r, model.variables)
  running
  running
  running
  running
  Alternatively, use the `__wrapped__` attribute to access the original
  model object.
  >>> grad = tape.gradient(r, model_recompute.__wrapped__.variables)
  running
  running
  running
  running
  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
  Returns:
    A function `g` wrapping `f` that defines a custom gradient, which recomputes
    `f` on the backwards pass of a gradient call.
  """
  # TODO(cdfreeman) Add is_recomputing functionality from graph mode version
  @custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = variable_scope.get_variable_scope()
    with record.stop_recording():
      result = f(*args, **kwargs)
    def grad_wrapper(*wrapper_args, variables=None):
      """Wrapper function to accomodate lack of kwargs in graph mode custom_gradient."""
      @custom_gradient
      def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        # Gradient calculation for reverse mode autodiff.
        with backprop.GradientTape() as t:
          id_args = nest.map_structure(gen_array_ops.identity, args)
          # Tuple `dresult` should contain at least one tensor.
          assert len(dresult) >= 1
          if not context.executing_eagerly():
            # XLA doesn't respect `tf.control_dependencies`. The code block
            # below manually adds a data dependency to `dresult` to ensure
            # recomputation of `f(*args, **kwargs)` happens after `dresult`.
            # This works even if `dresult[0]` is a size 0 tensor as reduce_max
            # of a size 0 tensor returns -inf. Use reshape here to avoid reading
            # the entire `dresult[0]`.
            elem = math_ops.reduce_max(array_ops.reshape(dresult[0], [-1])[:1])
            # Cast elem to bool in case elem is NaN.
            elem_bool = math_ops.cast(elem, dtypes.bool)
            dresult_dep = array_ops.where_v2(
                elem_bool == elem_bool, 0., float("nan"))  # pylint: disable=comparison-with-itself
            id_args = nest.map_structure(
                lambda x: x + math_ops.cast(dresult_dep, x.dtype), id_args)
          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with variable_scope.variable_scope(current_var_scope):
            recomputed_result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(
            recomputed_result,
            list(id_args) + kw_vars,
            output_gradients=dresult,
            unconnected_gradients=UnconnectedGradients.ZERO)
        def transpose(*t_args, **t_kwargs):
          """Gradient function calculation for forward mode autodiff."""
          # Just throw an error since gradients / activations are not stored on
          # tape for recompute.
          raise NotImplementedError(
              "recompute_grad tried to transpose grad of {}. "
              "Consider not using recompute_grad in forward mode"
              "autodiff".format(f.__name__))
        return (grads[:len(id_args)], grads[len(id_args):]), transpose
      return inner_recompute_grad(*wrapper_args)
    return result, grad_wrapper
  return tf_decorator.make_decorator(f, inner)
