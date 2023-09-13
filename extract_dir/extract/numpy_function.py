@tf_export("numpy_function")
@dispatch.add_dispatch_support
def numpy_function(func, inp, Tout, stateful=True, name=None):
  """Wraps a python function and uses it as a TensorFlow op.
  Given a python function `func` wrap this function as an operation in a
  TensorFlow function. `func` must take numpy arrays as its arguments and
  return numpy arrays as its outputs.
  The following example creates a TensorFlow graph with `np.sinh()` as an
  operation in the graph:
  >>> def my_numpy_func(x):
  ...   # x will be a numpy array with the contents of the input to the
  ...   # tf.function
  ...   return np.sinh(x)
  >>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
  ... def tf_function(input):
  ...   y = tf.numpy_function(my_numpy_func, [input], tf.float32)
  ...   return y * y
  >>> tf_function(tf.constant(1.))
  <tf.Tensor: shape=(), dtype=float32, numpy=1.3810978>
  Comparison to `tf.py_function`:
  `tf.py_function` and `tf.numpy_function` are very similar, except that
  `tf.numpy_function` takes numpy arrays, and not `tf.Tensor`s. If you want the
  function to contain `tf.Tensors`, and have any TensorFlow operations executed
  in the function be differentiable, please use `tf.py_function`.
  Note: We recommend to avoid using `tf.numpy_function` outside of
  prototyping and experimentation due to the following known limitations:
  * Calling `tf.numpy_function` will acquire the Python Global Interpreter Lock
    (GIL) that allows only one thread to run at any point in time. This will
    preclude efficient parallelization and distribution of the execution of the
    program. Therefore, you are discouraged to use `tf.numpy_function` outside
    of prototyping and experimentation.
  * The body of the function (i.e. `func`) will not be serialized in a
    `tf.SavedModel`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.
  * The operation must run in the same address space as the Python program
    that calls `tf.numpy_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.numpy_function`  you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).
  * Currently `tf.numpy_function` is not compatible with XLA. Calling
    `tf.numpy_function` inside `tf.function(jit_compile=True)` will raise an
    error.
  * Since the function takes numpy arrays, you cannot take gradients
    through a numpy_function. If you require something that is differentiable,
    please consider using tf.py_function.
  Args:
    func: A Python function, which accepts `numpy.ndarray` objects as arguments
      and returns a list of `numpy.ndarray` objects (or a single
      `numpy.ndarray`). This function must accept as many arguments as there are
      tensors in `inp`, and these argument types will match the corresponding
      `tf.Tensor` objects in `inp`. The returns `numpy.ndarray`s must match the
      number and types defined `Tout`.
      Important Note: Input and output `numpy.ndarray`s of `func` are not
        guaranteed to be copies. In some cases their underlying memory will be
        shared with the corresponding TensorFlow tensors. In-place modification
        or storing `func` input or return values in python datastructures
        without explicit (np.)copy can have non-deterministic consequences.
    inp: A list of `tf.Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) Setting this argument to False tells the runtime to
      treat the function as stateless, which enables certain optimizations.
      A function is stateless when given the same input it will return the
      same output and have no side effects; its only purpose is to have a
      return value.
      The behavior for a stateful function with the `stateful` argument False
      is undefined. In particular, caution should be taken when
      mutating the input arguments as this is a stateful operation.
    name: (Optional) A name for the operation.
  Returns:
    Single or list of `tf.Tensor` which `func` computes.
  """
  return py_func_common(func, inp, Tout, stateful=stateful, name=name)
