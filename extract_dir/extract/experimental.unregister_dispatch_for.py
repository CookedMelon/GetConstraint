@tf_export("experimental.unregister_dispatch_for")
def unregister_dispatch_for(dispatch_target):
  """Unregisters a function that was registered with `@dispatch_for_*`.
  This is primarily intended for testing purposes.
  Example:
  >>> # Define a type and register a dispatcher to override `tf.abs`:
  >>> class MyTensor(tf.experimental.ExtensionType):
  ...   value: tf.Tensor
  >>> @tf.experimental.dispatch_for_api(tf.abs)
  ... def my_abs(x: MyTensor):
  ...   return MyTensor(tf.abs(x.value))
  >>> tf.abs(MyTensor(5))
  MyTensor(value=<tf.Tensor: shape=(), dtype=int32, numpy=5>)
  >>> # Unregister the dispatcher, so `tf.abs` no longer calls `my_abs`.
  >>> unregister_dispatch_for(my_abs)
  >>> tf.abs(MyTensor(5))
  Traceback (most recent call last):
  ...
  ValueError: Attempt to convert a value ... to a Tensor.
  Args:
    dispatch_target: The function to unregister.
  Raises:
    ValueError: If `dispatch_target` was not registered using `@dispatch_for`,
      `@dispatch_for_unary_elementwise_apis`, or
      `@dispatch_for_binary_elementwise_apis`.
  """
  found = False
  # Check if dispatch_target registered by `@dispatch_for_api`
  for api, signatures in _TYPE_BASED_DISPATCH_SIGNATURES.items():
    if dispatch_target in signatures:
      dispatcher = getattr(api, TYPE_BASED_DISPATCH_ATTR)
      dispatcher.Unregister(dispatch_target)
      del signatures[dispatch_target]
      found = True
  # Check if dispatch_target registered by `@dispatch_for_*_elementwise_apis`
  elementwise_keys_to_delete = [
      key for (key, handler) in _ELEMENTWISE_API_HANDLERS.items()
      if handler is dispatch_target
  ]
  for key in set(elementwise_keys_to_delete):
    for _, target in _ELEMENTWISE_API_TARGETS[key]:
      unregister_dispatch_for(target)
    del _ELEMENTWISE_API_HANDLERS[key]
    del _ELEMENTWISE_API_TARGETS[key]
    found = True
  if not found:
    raise ValueError(f"Function {dispatch_target} was not registered using "
                     "a `@dispatch_for_*` decorator.")
