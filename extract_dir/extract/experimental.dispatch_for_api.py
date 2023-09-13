@tf_export("experimental.dispatch_for_api")
def dispatch_for_api(api, *signatures):
  """Decorator that overrides the default implementation for a TensorFlow API.
  The decorated function (known as the "dispatch target") will override the
  default implementation for the API when the API is called with parameters that
  match a specified type signature.  Signatures are specified using dictionaries
  that map parameter names to type annotations.  E.g., in the following example,
  `masked_add` will be called for `tf.add` if both `x` and `y` are
  `MaskedTensor`s:
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor, 'y': MaskedTensor})
  ... def masked_add(x, y, name=None):
  ...   return MaskedTensor(x.values + y.values, x.mask & y.mask)
  >>> mt = tf.add(MaskedTensor([1, 2], [True, False]), MaskedTensor(10, True))
  >>> print(f"values={mt.values.numpy()}, mask={mt.mask.numpy()}")
  values=[11 12], mask=[ True False]
  If multiple type signatures are specified, then the dispatch target will be
  called if any of the signatures match.  For example, the following code
  registers `masked_add` to be called if `x` is a `MaskedTensor` *or* `y` is
  a `MaskedTensor`.
  >>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor}, {'y':MaskedTensor})
  ... def masked_add(x, y):
  ...   x_values = x.values if isinstance(x, MaskedTensor) else x
  ...   x_mask = x.mask if isinstance(x, MaskedTensor) else True
  ...   y_values = y.values if isinstance(y, MaskedTensor) else y
  ...   y_mask = y.mask if isinstance(y, MaskedTensor) else True
  ...   return MaskedTensor(x_values + y_values, x_mask & y_mask)
  The type annotations in type signatures may be type objects (e.g.,
  `MaskedTensor`), `typing.List` values, or `typing.Union` values.   For
  example, the following will register `masked_concat` to be called if `values`
  is a list of `MaskedTensor` values:
  >>> @dispatch_for_api(tf.concat, {'values': typing.List[MaskedTensor]})
  ... def masked_concat(values, axis):
  ...   return MaskedTensor(tf.concat([v.values for v in values], axis),
  ...                       tf.concat([v.mask for v in values], axis))
  Each type signature must contain at least one subclass of `tf.CompositeTensor`
  (which includes subclasses of `tf.ExtensionType`), and dispatch will only be
  triggered if at least one type-annotated parameter contains a
  `CompositeTensor` value.  This rule avoids invoking dispatch in degenerate
  cases, such as the following examples:
  * `@dispatch_for_api(tf.concat, {'values': List[MaskedTensor]})`: Will not
    dispatch to the decorated dispatch target when the user calls
    `tf.concat([])`.
  * `@dispatch_for_api(tf.add, {'x': Union[MaskedTensor, Tensor], 'y':
    Union[MaskedTensor, Tensor]})`: Will not dispatch to the decorated dispatch
    target when the user calls `tf.add(tf.constant(1), tf.constant(2))`.
  The dispatch target's signature must match the signature of the API that is
  being overridden.  In particular, parameters must have the same names, and
  must occur in the same order.  The dispatch target may optionally elide the
  "name" parameter, in which case it will be wrapped with a call to
  `tf.name_scope` when appropraite.
  Args:
    api: The TensorFlow API to override.
    *signatures: Dictionaries mapping parameter names or indices to type
      annotations, specifying when the dispatch target should be called.  In
      particular, the dispatch target will be called if any signature matches;
      and a signature matches if all of the specified parameters have types that
      match with the indicated type annotations.  If no signatures are
      specified, then a signature will be read from the dispatch target
      function's type annotations.
  Returns:
    A decorator that overrides the default implementation for `api`.
  #### Registered APIs
  The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:
  <<API_LIST>>
  """
  dispatcher = getattr(api, TYPE_BASED_DISPATCH_ATTR, None)
  if dispatcher is None:
    raise ValueError(f"{api} does not support dispatch.")
  api_signature = tf_inspect.signature(api)
  signature_checkers = [
      _make_signature_checker(api_signature, signature)
      for signature in signatures
  ]
  def decorator(dispatch_target):
    """Decorator that registers the given dispatch target."""
    if not callable(dispatch_target):
      raise TypeError("Expected dispatch_target to be callable; "
                      f"got {dispatch_target!r}")
    dispatch_target = _add_name_scope_wrapper(dispatch_target, api_signature)
    _check_signature(api_signature, dispatch_target)
    for signature_checker in signature_checkers:
      dispatcher.Register(signature_checker, dispatch_target)
    _TYPE_BASED_DISPATCH_SIGNATURES[api][dispatch_target].extend(signatures)
    if not signature_checkers:
      signature = _signature_from_annotations(dispatch_target)
      checker = _make_signature_checker(api_signature, signature)
      dispatcher.Register(checker, dispatch_target)
      _TYPE_BASED_DISPATCH_SIGNATURES[api][dispatch_target].append(signature)
    return dispatch_target
  return decorator
