@tf_export("experimental.dispatch_for_binary_elementwise_assert_apis")
def dispatch_for_binary_elementwise_assert_apis(x_type, y_type):
  """Decorator to override default implementation for binary elementwise assert APIs.
  The decorated function (known as the "elementwise assert handler")
  overrides the default implementation for any binary elementwise assert API
  whenever the value for the first two arguments (typically named `x` and `y`)
  match the specified type annotations.  The handler is called with two
  arguments:
    `elementwise_assert_handler(assert_func, x, y)`
  Where `x` and `y` are the first two arguments to the binary elementwise assert
  operation, and `assert_func` is a TensorFlow function that takes two
  parameters and performs the elementwise assert operation (e.g.,
  `tf.debugging.assert_equal`).
  The following example shows how this decorator can be used to update all
  binary elementwise assert operations to handle a `MaskedTensor` type:
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_binary_elementwise_assert_apis(MaskedTensor, MaskedTensor)
  ... def binary_elementwise_assert_api_handler(assert_func, x, y):
  ...   merged_mask = tf.logical_and(x.mask, y.mask)
  ...   selected_x_values = tf.boolean_mask(x.values, merged_mask)
  ...   selected_y_values = tf.boolean_mask(y.values, merged_mask)
  ...   assert_func(selected_x_values, selected_y_values)
  >>> a = MaskedTensor([1, 1, 0, 1, 1], [False, False, True, True, True])
  >>> b = MaskedTensor([2, 2, 0, 2, 2], [True, True, True, False, False])
  >>> tf.debugging.assert_equal(a, b) # assert passed; no exception was thrown
  >>> a = MaskedTensor([1, 1, 1, 1, 1], [True, True, True, True, True])
  >>> b = MaskedTensor([0, 0, 0, 0, 2], [True, True, True, True, True])
  >>> tf.debugging.assert_greater(a, b)
  Traceback (most recent call last):
  ...
  InvalidArgumentError: Condition x > y did not hold.
  Args:
    x_type: A type annotation indicating when the api handler should be called.
    y_type: A type annotation indicating when the api handler should be called.
  Returns:
    A decorator.
  #### Registered APIs
  The binary elementwise assert APIs are:
  <<API_LIST>>
  """
  def decorator(handler):
    api_handler_key = (x_type, y_type, _ASSERT_API_TAG)
    if api_handler_key in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A binary elementwise assert dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[api_handler_key]}) "
                       f"has already been registered for ({x_type}, {y_type}).")
    _ELEMENTWISE_API_HANDLERS[api_handler_key] = handler
    for api in _BINARY_ELEMENTWISE_ASSERT_APIS:
      _add_dispatch_for_binary_elementwise_api(api, x_type, y_type, handler)
    return handler
  return decorator
