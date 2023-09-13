@tf_export("py_function")
@dispatch.add_dispatch_support
def eager_py_func(func, inp, Tout, name=None):
  """Wraps a python function into a TensorFlow op that executes it eagerly.
  This function allows expressing computations in a TensorFlow graph as
  Python functions. In particular, it wraps a Python function `func`
  in a once-differentiable TensorFlow operation that executes it with eager
  execution enabled. As a consequence, `tf.py_function` makes it
  possible to express control flow using Python constructs (`if`, `while`,
  `for`, etc.), instead of TensorFlow control flow constructs (`tf.cond`,
  `tf.while_loop`). For example, you might use `tf.py_function` to
  implement the log huber function:
  ```python
  def log_huber(x, m):
    if tf.abs(x) <= m:
      return x**2
    else:
      return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))
  x = tf.constant(1.0)
  m = tf.constant(2.0)
  with tf.GradientTape() as t:
    t.watch([x, m])
    y = tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32)
  dy_dx = t.gradient(y, x)
  assert dy_dx.numpy() == 2.0
  ```
  You can also use `tf.py_function` to debug your models at runtime
  using Python tools, i.e., you can isolate portions of your code that
  you want to debug, wrap them in Python functions and insert `pdb` tracepoints
  or print statements as desired, and wrap those functions in
  `tf.py_function`.
  For more information on eager execution, see the
  [Eager guide](https://tensorflow.org/guide/eager).
  `tf.py_function` is similar in spirit to `tf.compat.v1.py_func`, but unlike
  the latter, the former lets you use TensorFlow operations in the wrapped
  Python function. In particular, while `tf.compat.v1.py_func` only runs on CPUs
  and wraps functions that take NumPy arrays as inputs and return NumPy arrays
  as outputs, `tf.py_function` can be placed on GPUs and wraps functions
  that take Tensors as inputs, execute TensorFlow operations in their bodies,
  and return Tensors as outputs.
  Note: We recommend to avoid using `tf.py_function` outside of prototyping
  and experimentation due to the following known limitations:
  * Calling `tf.py_function` will acquire the Python Global Interpreter Lock
    (GIL) that allows only one thread to run at any point in time. This will
    preclude efficient parallelization and distribution of the execution of the
    program.
  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.
  * The operation must run in the same address space as the Python program
    that calls `tf.py_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.py_function()` and you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).
  * Currently `tf.py_function` is not compatible with XLA. Calling
    `tf.py_function` inside `tf.function(jit_compile=True)` will raise an
    error.
  Args:
    func: A Python function that accepts `inp` as arguments, and returns a
      value (or list of values) whose type is described by `Tout`.
    inp: Input arguments for `func`.  A list whose elements are `Tensor`s or
      `CompositeTensors` (such as `tf.RaggedTensor`); or a single `Tensor` or
      `CompositeTensor`.
    Tout: The type(s) of the value(s) returned by `func`.  One of the
      following.
      * If `func` returns a `Tensor` (or a value that can be converted to a
        Tensor): the `tf.DType` for that value.
      * If `func` returns a `CompositeTensor`: The `tf.TypeSpec` for that value.
      * If `func` returns `None`: the empty list (`[]`).
      * If `func` returns a list of `Tensor` and `CompositeTensor` values:
        a corresponding list of `tf.DType`s and `tf.TypeSpec`s for each value.
    name: A name for the operation (optional).
  Returns:
    The value(s) computed by `func`: a `Tensor`, `CompositeTensor`, or list of
    `Tensor` and `CompositeTensor`; or an empty list if `func` returns `None`.
  """
  if ops.executing_eagerly_outside_functions():
    with ops.device(context.context().host_address_space()):
      return _internal_py_func(
          func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)
  return _internal_py_func(
      func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)
