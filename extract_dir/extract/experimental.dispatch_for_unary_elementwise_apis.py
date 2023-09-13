@tf_export("experimental.dispatch_for_unary_elementwise_apis")
def dispatch_for_unary_elementwise_apis(x_type):
  """Decorator to override default implementation for unary elementwise APIs.
  The decorated function (known as the "elementwise api handler") overrides
  the default implementation for any unary elementwise API whenever the value
  for the first argument (typically named `x`) matches the type annotation
  `x_type`. The elementwise api handler is called with two arguments:
    `elementwise_api_handler(api_func, x)`
  Where `api_func` is a function that takes a single parameter and performs the
  elementwise operation (e.g., `tf.abs`), and `x` is the first argument to the
  elementwise api.
  The following example shows how this decorator can be used to update all
  unary elementwise operations to handle a `MaskedTensor` type:
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_unary_elementwise_apis(MaskedTensor)
  ... def unary_elementwise_api_handler(api_func, x):
  ...   return MaskedTensor(api_func(x.values), x.mask)
  >>> mt = MaskedTensor([1, -2, -3], [True, False, True])
  >>> abs_mt = tf.abs(mt)
  >>> print(f"values={abs_mt.values.numpy()}, mask={abs_mt.mask.numpy()}")
  values=[1 2 3], mask=[ True False True]
  For unary elementwise operations that take extra arguments beyond `x`, those
  arguments are *not* passed to the elementwise api handler, but are
  automatically added when `api_func` is called.  E.g., in the following
  example, the `dtype` parameter is not passed to
  `unary_elementwise_api_handler`, but is added by `api_func`.
  >>> ones_mt = tf.ones_like(mt, dtype=tf.float32)
  >>> print(f"values={ones_mt.values.numpy()}, mask={ones_mt.mask.numpy()}")
  values=[1.0 1.0 1.0], mask=[ True False True]
  Args:
    x_type: A type annotation indicating when the api handler should be called.
      See `dispatch_for_api` for a list of supported annotation types.
  Returns:
    A decorator.
  #### Registered APIs
  The unary elementwise APIs are:
  <<API_LIST>>
  """
  def decorator(handler):
    if (x_type,) in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A unary elementwise dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[(x_type,)]}) "
                       f"has already been registered for {x_type}.")
    _ELEMENTWISE_API_HANDLERS[(x_type,)] = handler
    for api in _UNARY_ELEMENTWISE_APIS:
      _add_dispatch_for_unary_elementwise_api(api, x_type, handler)
    return handler
  return decorator
