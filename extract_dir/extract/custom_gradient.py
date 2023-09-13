@tf_export("custom_gradient")
def custom_gradient(f=None):
  """Decorator to define a function with a custom gradient.
  This decorator allows fine grained control over the gradients of a sequence
  for operations.  This may be useful for multiple reasons, including providing
  a more efficient or numerically stable gradient for a sequence of operations.
  For example, consider the following function that commonly occurs in the
  computation of cross entropy and log likelihoods:
  ```python
  def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))
  ```
  Due to numerical instability, the gradient of this function evaluated at x=100
  is NaN.  For example:
  ```python
  with tf.GradientTape() as tape:
    tape.watch(x)
    y=log1pexp(x)
  dy_dx = tape.gradient(y, x) # Will be NaN when evaluated.
  ```
  The gradient expression can be analytically simplified to provide numerical
  stability:
  ```python
  @tf.custom_gradient
  def log1pexp(x):
    e = tf.exp(x)
    def grad(upstream):
      return upstream * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad
  ```
  With this definition, the gradient `dy_dx` at `x = 100` will be correctly
  evaluated as 1.0.
  The variable `upstream` is defined as the upstream gradient. i.e. the gradient
  from all the layers or functions originating from this layer. The above
  example has no upstream functions, therefore `upstream = dy/dy = 1.0`.
  Assume that `x_i` is `log1pexp` in the forward pass `x_1 = x_1(x_0)`,
  `x_2 = x_2(x_1)`, ..., `x_i = x_i(x_i-1)`, ..., `x_n = x_n(x_n-1)`. By
  chain rule we know that `dx_n/dx_0 = dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... *
  dx_i/dx_i-1 * ... * dx_1/dx_0`.
  In this case the gradient of our current function defined as
  `dx_i/dx_i-1 = (1 - 1 / (1 + e))`. The upstream gradient `upstream` would be
  `dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... * dx_i+1/dx_i`. The upstream gradient
  multiplied by the current gradient is then passed downstream.
  In case the function takes multiple variables as input, the `grad`
  function must also return  the same number of variables.
  We take the function `z = x * y` as an example.
  >>> @tf.custom_gradient
  ... def bar(x, y):
  ...   def grad(upstream):
  ...     dz_dx = y
  ...     dz_dy = x
  ...     return upstream * dz_dx, upstream * dz_dy
  ...   z = x * y
  ...   return z, grad
  >>> x = tf.constant(2.0, dtype=tf.float32)
  >>> y = tf.constant(3.0, dtype=tf.float32)
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   tape.watch(x)
  ...   tape.watch(y)
  ...   z = bar(x, y)
  >>> z
  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  >>> tape.gradient(z, x)
  <tf.Tensor: shape=(), dtype=float32, numpy=3.0>
  >>> tape.gradient(z, y)
  <tf.Tensor: shape=(), dtype=float32, numpy=2.0>
  Nesting custom gradients can lead to unintuitive results. The default
  behavior does not correspond to n-th order derivatives. For example
  ```python
  @tf.custom_gradient
  def op(x):
    y = op1(x)
    @tf.custom_gradient
    def grad_fn(dy):
      gdy = op2(x, y, dy)
      def grad_grad_fn(ddy):  # Not the 2nd order gradient of op w.r.t. x.
        return op3(x, y, dy, ddy)
      return gdy, grad_grad_fn
    return y, grad_fn
  ```
  The function `grad_grad_fn` will be calculating the first order gradient
  of `grad_fn` with respect to `dy`, which is used to generate forward-mode
  gradient graphs from backward-mode gradient graphs, but is not the same as
  the second order gradient of `op` with respect to `x`.
  Instead, wrap nested `@tf.custom_gradients` in another function:
  ```python
  @tf.custom_gradient
  def op_with_fused_backprop(x):
    y, x_grad = fused_op(x)
    def first_order_gradient(dy):
      @tf.custom_gradient
      def first_order_custom(unused_x):
        def second_order_and_transpose(ddy):
          return second_order_for_x(...), gradient_wrt_dy(...)
        return x_grad, second_order_and_transpose
      return dy * first_order_custom(x)
    return y, first_order_gradient
  ```
  Additional arguments to the inner `@tf.custom_gradient`-decorated function
  control the expected return values of the innermost function.
  The examples above illustrate how to specify custom gradients for functions
  which do not read from variables. The following example uses variables, which
  require special handling because they are effectively inputs of the forward
  function.
  >>> weights = tf.Variable(tf.ones([2]))  # Trainable variable weights
  >>> @tf.custom_gradient
  ... def linear_poly(x):
  ...   # Creating polynomial
  ...   poly = weights[1] * x + weights[0]
  ...
  ...   def grad_fn(dpoly, variables):
  ...     # dy/dx = weights[1] and we need to left multiply dpoly
  ...     grad_xs = dpoly * weights[1]  # Scalar gradient
  ...
  ...     grad_vars = []  # To store gradients of passed variables
  ...     assert variables is not None
  ...     assert len(variables) == 1
  ...     assert variables[0] is weights
  ...     # Manually computing dy/dweights
  ...     dy_dw = dpoly * tf.stack([x ** 1, x ** 0])
  ...     grad_vars.append(
  ...         tf.reduce_sum(tf.reshape(dy_dw, [2, -1]), axis=1)
  ...     )
  ...     return grad_xs, grad_vars
  ...   return poly, grad_fn
  >>> x = tf.constant([1., 2., 3.])
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   tape.watch(x)
  ...   poly = linear_poly(x)
  >>> poly # poly = x + 1
  <tf.Tensor: shape=(3,),
    dtype=float32,
    numpy=array([2., 3., 4.], dtype=float32)>
  >>> tape.gradient(poly, x)  # conventional scalar gradient dy/dx
  <tf.Tensor: shape=(3,),
    dtype=float32,
    numpy=array([1., 1., 1.], dtype=float32)>
  >>> tape.gradient(poly, weights)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 3.], dtype=float32)>
  Above example illustrates usage of trainable variable `weights`.
  In the example, the inner `grad_fn` accepts an extra `variables` input
  parameter and also returns an extra `grad_vars` output. That extra argument
  is passed if the forward function reads any variables. You need to
  compute the gradient w.r.t. each of those `variables` and output it as a list
  of `grad_vars`. Note here that default value of `variables` is set to `None`
  when no variables are used in the forward function.
  It should be noted `tf.GradientTape` is still watching the forward pass of a
  `tf.custom_gradient`, and will use the ops it watches. As a consequence,
  calling `tf.function` while the tape is still watching leads
  to a gradient graph being built. If an op is used in `tf.function` without
  registered gradient, a `LookupError` will be raised.
  Users can insert `tf.stop_gradient` to customize this behavior. This
  is demonstrated in the example below. `tf.random.shuffle` does not have a
  registered gradient. As a result `tf.stop_gradient` is used to avoid the
  `LookupError`.
  ```python
  x = tf.constant([0.3, 0.5], dtype=tf.float32)
  @tf.custom_gradient
  def test_func_with_stop_grad(x):
    @tf.function
    def _inner_func():
      # Avoid exception during the forward pass
      return tf.stop_gradient(tf.random.shuffle(x))
      # return tf.random.shuffle(x)  # This will raise
    res = _inner_func()
    def grad(upstream):
      return upstream  # Arbitrarily defined custom gradient
    return res, grad
  with tf.GradientTape() as g:
    g.watch(x)
    res = test_func_with_stop_grad(x)
  g.gradient(res, x)
  ```
  See also `tf.RegisterGradient` which registers a gradient function for a
  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows
  for fine grained control over the gradient computation of a sequence of
  operations.
  Note that if the decorated function uses `Variable`s, the enclosing variable
  scope must be using
  [ResourceVariables](https://www.tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables).
  Args:
    f: function `f(*x)` that returns a tuple `(y, grad_fn)` where:
       - `x` is a sequence of (nested structures of) `Tensor` inputs to the
         function.
       - `y` is a (nested structure of) `Tensor` outputs of applying TensorFlow
         operations in `f` to `x`.
       - `grad_fn` is a function with the signature `g(*grad_ys)` which returns
         a list of `Tensor`s the same size as (flattened) `x` - the derivatives
         of `Tensor`s in `y` with respect to the `Tensor`s in `x`.  `grad_ys` is
         a sequence of `Tensor`s the same size as (flattened) `y` holding the
         initial value gradients for each `Tensor` in `y`.
         In a pure mathematical sense, a vector-argument vector-valued function
         `f`'s derivatives should be its Jacobian matrix `J`. Here we are
         expressing the Jacobian `J` as a function `grad_fn` which defines how
         `J` will transform a vector `grad_ys` when left-multiplied with it
         (`grad_ys * J`, the vector-Jacobian product, or VJP). This functional
         representation of a matrix is convenient to use for chain-rule
         calculation (in e.g. the back-propagation algorithm).
         If `f` uses `Variable`s (that are not part of the
         inputs), i.e. through `get_variable`, then `grad_fn` should have
         signature `g(*grad_ys, variables=None)`, where `variables` is a list of
         the `Variable`s, and return a 2-tuple `(grad_xs, grad_vars)`, where
         `grad_xs` is the same as above, and `grad_vars` is a `list<Tensor>`
         with the derivatives of `Tensor`s in `y` with respect to the variables
         (that is, grad_vars has one Tensor per variable in variables).
  Returns:
    A function `h(x)` which returns the same value as `f(x)[0]` and whose
    gradient (as calculated by `tf.gradients`) is determined by `f(x)[1]`.
  """
  if f is None:
    return lambda f: custom_gradient(f=f)
  @Bind.decorator
  def decorated(wrapped, args, kwargs):
    """Decorated function with custom gradient."""
    if context.executing_eagerly():
      return _eager_mode_decorator(wrapped, args, kwargs)
    else:
      return _graph_mode_decorator(wrapped, args, kwargs)
  return tf_decorator.make_decorator(f, decorated(f))  # pylint: disable=no-value-for-parameter
