@tf_export("experimental.dispatch_for_binary_elementwise_apis")
def dispatch_for_binary_elementwise_apis(x_type, y_type):
  """Decorator to override default implementation for binary elementwise APIs.
  The decorated function (known as the "elementwise api handler") overrides
  the default implementation for any binary elementwise API whenever the value
  for the first two arguments (typically named `x` and `y`) match the specified
  type annotations.  The elementwise api handler is called with two arguments:
    `elementwise_api_handler(api_func, x, y)`
  Where `x` and `y` are the first two arguments to the elementwise api, and
  `api_func` is a TensorFlow function that takes two parameters and performs the
  elementwise operation (e.g., `tf.add`).
  The following example shows how this decorator can be used to update all
  binary elementwise operations to handle a `MaskedTensor` type:
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
  ... def binary_elementwise_api_handler(api_func, x, y):
  ...   return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
  >>> a = MaskedTensor([1, 2, 3, 4, 5], [True, True, True, True, False])
  >>> b = MaskedTensor([2, 4, 6, 8, 0], [True, True, True, False, True])
  >>> c = tf.add(a, b)
  >>> print(f"values={c.values.numpy()}, mask={c.mask.numpy()}")
  values=[ 3 6 9 12 5], mask=[ True True True False False]
  Args:
    x_type: A type annotation indicating when the api handler should be called.
    y_type: A type annotation indicating when the api handler should be called.
  Returns:
    A decorator.
  #### Registered APIs
  The binary elementwise APIs are:
  <<API_LIST>>
  """
  def decorator(handler):
    if (x_type, y_type) in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A binary elementwise dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[x_type, y_type]}) "
                       f"has already been registered for ({x_type}, {y_type}).")
    _ELEMENTWISE_API_HANDLERS[x_type, y_type] = handler
    for api in _BINARY_ELEMENTWISE_APIS:
      _add_dispatch_for_binary_elementwise_api(api, x_type, y_type, handler)
    return handler
  return decorator
