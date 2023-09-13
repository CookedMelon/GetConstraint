"/home/cc/Workspace/tfconstraint/python/ops/check_ops.py"
@tf_export(
    'debugging.assert_proper_iterable',
    v1=['debugging.assert_proper_iterable', 'assert_proper_iterable'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_proper_iterable')
def assert_proper_iterable(values):
  """Static assert that values is a "proper" iterable.
  `Ops` that expect iterables of `Tensor` can call this to validate input.
  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.
  Args:
    values:  Object to be checked.
  Raises:
    TypeError:  If `values` is not iterable or is one of
      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
  """
  unintentional_iterables = (
      (ops.Tensor, sparse_tensor.SparseTensor, np.ndarray)
      + compat.bytes_or_text_types
  )
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))
  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))
@tf_export('debugging.assert_negative', v1=[])
@dispatch.add_dispatch_support
def assert_negative_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x < 0` holds element-wise.
  This Op checks that `x[i] < 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.
  If `x` is not negative everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.
  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_negative".
  Returns:
    Op raising `InvalidArgumentError` unless `x` is all negative. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] < 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_negative(x=x, message=message, summarize=summarize, name=name)
@tf_export(v1=['debugging.assert_negative', 'assert_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_negative')
@_unary_assert_doc('< 0', 'negative')
def assert_negative(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x < 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(x, zero, data=data, summarize=summarize)
@tf_export('debugging.assert_positive', v1=[])
@dispatch.add_dispatch_support
def assert_positive_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x > 0` holds element-wise.
  This Op checks that `x[i] > 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.
  If `x` is not positive everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.
  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional). Defaults to "assert_positive".
  Returns:
    Op raising `InvalidArgumentError` unless `x` is all positive. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] > 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_positive(x=x, summarize=summarize, message=message, name=name)
@tf_export(v1=['debugging.assert_positive', 'assert_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_positive')
@_unary_assert_doc('> 0', 'positive')
def assert_positive(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message, 'Condition x > 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(zero, x, data=data, summarize=summarize)
@tf_export('debugging.assert_non_negative', v1=[])
@dispatch.add_dispatch_support
def assert_non_negative_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x >= 0` holds element-wise.
  This Op checks that `x[i] >= 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.
  If `x` is not >= 0 everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.
  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to
      "assert_non_negative".
  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative. This can
      be used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] >= 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_non_negative(x=x, summarize=summarize, message=message,
                             name=name)
@tf_export(v1=['debugging.assert_non_negative', 'assert_non_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_negative')
@_unary_assert_doc('>= 0', 'non-negative')
def assert_non_negative(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_non_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x >= 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(zero, x, data=data, summarize=summarize)
@tf_export('debugging.assert_non_positive', v1=[])
@dispatch.add_dispatch_support
def assert_non_positive_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x <= 0` holds element-wise.
  This Op checks that `x[i] <= 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.
  If `x` is not <= 0 everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.
  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to
      "assert_non_positive".
  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-positive. This can
      be used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] <= 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_non_positive(x=x, summarize=summarize, message=message,
                             name=name)
@tf_export(v1=['debugging.assert_non_positive', 'assert_non_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_positive')
@_unary_assert_doc('<= 0', 'non-positive')
def assert_non_positive(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_non_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x <= 0 did not hold element-wise:'
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(x, zero, data=data, summarize=summarize)
@tf_export('debugging.assert_equal', 'assert_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('==', 'assert_equal', 3)
def assert_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_equal(x=x, y=y, summarize=summarize, message=message, name=name)
@tf_export(v1=['debugging.assert_equal', 'assert_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('==', '[1, 2]')
def assert_equal(x, y, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  with ops.name_scope(name, 'assert_equal', [x, y, data]):
    # Short-circuit if x and y are the same tensor.
    if x is y:
      return None if context.executing_eagerly() else control_flow_ops.no_op()
  return _binary_assert('==', 'assert_equal', math_ops.equal, np.equal, x, y,
                        data, summarize, message, name)
@tf_export('debugging.assert_none_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('!=', 'assert_none_equal', 6)
def assert_none_equal_v2(x, y, summarize=None, message=None, name=None):
  return assert_none_equal(x=x, y=y, summarize=summarize, message=message,
                           name=name)
@tf_export(v1=['debugging.assert_none_equal', 'assert_none_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_none_equal')
@_binary_assert_doc('!=', '[2, 1]')
def assert_none_equal(
    x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('!=', 'assert_none_equal', math_ops.not_equal,
                        np.not_equal, x, y, data, summarize, message, name)
@tf_export('debugging.assert_near', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
def assert_near_v2(x, y, rtol=None, atol=None, message=None, summarize=None,
                   name=None):
  """Assert the condition `x` and `y` are close element-wise.
  This Op checks that `x[i] - y[i] < atol + rtol * tf.abs(y[i])` holds for every
  pair of (possibly broadcast) elements of `x` and `y`. If both `x` and `y` are
  empty, this is trivially satisfied.
  If any elements of `x` and `y` are not close, `message`, as well as the first
  `summarize` entries of `x` and `y` are printed, and `InvalidArgumentError`
  is raised.
  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.
  Args:
    x: Float or complex `Tensor`.
    y: Float or complex `Tensor`, same dtype as and broadcastable to `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_near".
  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.
      This can be used with `tf.control_dependencies` inside of `tf.function`s
      to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x != y` is False for any pair of elements in `x` and `y`. The check can
      be performed immediately during eager execution or if `x` and `y` are
      statically known.
  @compatibility(numpy)
  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data
  type. This is due to the fact that `TensorFlow` is often used with `32bit`,
  `64bit`, and even `16bit` data.
  @end_compatibility
  """
  return assert_near(x=x, y=y, rtol=rtol, atol=atol, summarize=summarize,
                     message=message, name=name)
@tf_export(v1=['debugging.assert_near', 'assert_near'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_near')
def assert_near(
    x, y, rtol=None, atol=None, data=None, summarize=None, message=None,
    name=None):
  """Assert the condition `x` and `y` are close element-wise.
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.compat.v1.assert_near(x, y)]):
    output = tf.reduce_sum(x)
  ```
  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have
  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.
  If both `x` and `y` are empty, this is trivially satisfied.
  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.
  Args:
    x:  Float or complex `Tensor`.
    y:  Float or complex `Tensor`, same `dtype` as, and broadcastable to, `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_near".
  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.
  @compatibility(numpy)
  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data
  type. This is due to the fact that `TensorFlow` is often used with `32bit`,
  `64bit`, and even `16bit` data.
  @end_compatibility
  """
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_near', [x, y, rtol, atol, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y', dtype=x.dtype)
    dtype = x.dtype
    if dtype.is_complex:
      dtype = dtype.real_dtype
    eps = np.finfo(dtype.as_numpy_dtype).eps
    rtol = 10 * eps if rtol is None else rtol
    atol = 10 * eps if atol is None else atol
    rtol = ops.convert_to_tensor(rtol, name='rtol', dtype=dtype)
    atol = ops.convert_to_tensor(atol, name='atol', dtype=dtype)
    if context.executing_eagerly():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name
    if data is None:
      data = [
          message,
          'x and y not equal to tolerance rtol = %s, atol = %s' % (rtol, atol),
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    tol = atol + rtol * math_ops.abs(y)
    diff = math_ops.abs(x - y)
    condition = math_ops.reduce_all(math_ops.less(diff, tol))
    return control_flow_assert.Assert(condition, data, summarize=summarize)
@tf_export('debugging.assert_less', 'assert_less', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<', 'assert_less', 3)
def assert_less_v2(x, y, message=None, summarize=None, name=None):
  return assert_less(x=x, y=y, summarize=summarize, message=message, name=name)
@tf_export(v1=['debugging.assert_less', 'assert_less'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('<', '[2, 3]')
def assert_less(x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('<', 'assert_less', math_ops.less, np.less, x, y, data,
                        summarize, message, name)
@tf_export('debugging.assert_less_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<=', 'assert_less_equal', 3)
def assert_less_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_less_equal(x=x, y=y,
                           summarize=summarize, message=message, name=name)
@tf_export(v1=['debugging.assert_less_equal', 'assert_less_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_less_equal')
@_binary_assert_doc('<=', '[1, 3]')
def assert_less_equal(x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('<=', 'assert_less_equal', math_ops.less_equal,
                        np.less_equal, x, y, data, summarize, message, name)
@tf_export('debugging.assert_greater', 'assert_greater', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>', 'assert_greater', 9)
def assert_greater_v2(x, y, message=None, summarize=None, name=None):
  return assert_greater(x=x, y=y, summarize=summarize, message=message,
                        name=name)
@tf_export(v1=['debugging.assert_greater', 'assert_greater'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('>', '[0, 1]')
def assert_greater(x, y, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  return _binary_assert('>', 'assert_greater', math_ops.greater, np.greater, x,
                        y, data, summarize, message, name)
@tf_export('debugging.assert_greater_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>=', 'assert_greater_equal', 9)
def assert_greater_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_greater_equal(x=x, y=y, summarize=summarize, message=message,
                              name=name)
@tf_export(v1=['debugging.assert_greater_equal', 'assert_greater_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_greater_equal')
@_binary_assert_doc('>=', '[1, 0]')
def assert_greater_equal(x, y, data=None, summarize=None, message=None,
                         name=None):
  return _binary_assert('>=', 'assert_greater_equal', math_ops.greater_equal,
                        np.greater_equal, x, y, data, summarize, message, name)
def _assert_rank_condition(
    x, rank, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.
  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    static_condition:   A python function that takes `[actual_rank, given_rank]`
      and returns `True` if the condition is satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_rank] and return
      `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.
  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  assert_type(rank, dtypes.int32)
  # Attempt to statically defined rank.
  rank_static = tensor_util.constant_value(rank)
  if rank_static is not None:
    if rank_static.ndim != 0:
      raise ValueError('Rank must be a scalar.')
    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, rank_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, rank_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')
  condition = dynamic_condition(array_ops.rank(x), rank)
  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  if rank_static is None:
    this_data = ['Rank must be a scalar. Received rank: ', rank]
    rank_check = assert_rank(rank, 0, data=this_data)
    condition = control_flow_ops.with_dependencies([rank_check], condition)
  return control_flow_assert.Assert(condition, data, summarize=summarize)
@tf_export('debugging.assert_rank', 'assert_rank', v1=[])
@dispatch.add_dispatch_support
def assert_rank_v2(x, rank, message=None, name=None):
  """Assert that `x` has rank equal to `rank`.
  This Op checks that the rank of `x` is equal to `rank`.
  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.
  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to
      "assert_rank".
  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x` does not have rank `rank`. The check can be performed immediately
      during eager execution or if the shape of `x` is statically known.
  """
  return assert_rank(x=x, rank=rank, message=message, name=name)
@tf_export(v1=['debugging.assert_rank', 'assert_rank'])
@dispatch.add_dispatch_support
def assert_rank(x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank`.
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```
  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and the shape of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_rank".
  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.
  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(name, 'assert_rank', (x, rank) + tuple(data or [])):
    if not isinstance(x, sparse_tensor.SparseTensor):
      x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = _message_prefix(message)
    static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
    dynamic_condition = math_ops.equal
    if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
      name = ''
    else:
      name = x.name
    if data is None:
      data = [
          message,
          'Tensor %s must have rank' % name, rank, 'Received shape: ',
          array_ops.shape(x)
      ]
    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)
    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank %d.  Received rank %d, shape %s' %
            (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise ValueError(e.args[0])
  return assert_op
@tf_export('debugging.assert_rank_at_least', v1=[])
@dispatch.add_dispatch_support
def assert_rank_at_least_v2(x, rank, message=None, name=None):
  """Assert that `x` has rank of at least `rank`.
  This Op checks that the rank of `x` is greater or equal to `rank`.
  If `x` has a rank lower than `rank`, `message`, as well as the shape of `x`
  are printed, and `InvalidArgumentError` is raised.
  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to
      "assert_rank_at_least".
  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: `x` does not have rank at least `rank`, but the rank
      cannot be statically determined.
    ValueError: If static checks determine `x` has mismatched rank.
  """
  return assert_rank_at_least(x=x, rank=rank, message=message, name=name)
@tf_export(v1=['debugging.assert_rank_at_least', 'assert_rank_at_least'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_at_least')
def assert_rank_at_least(
    x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```
  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".
  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.
  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(
      name, 'assert_rank_at_least', (x, rank) + tuple(data or [])):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = _message_prefix(message)
    static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
    dynamic_condition = math_ops.greater_equal
    if context.executing_eagerly():
      name = ''
    else:
      name = x.name
    if data is None:
      data = [
          message,
          'Tensor %s must have rank at least' % name, rank,
          'Received shape: ', array_ops.shape(x)
      ]
    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)
    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank at least %d.  Received rank %d, '
            'shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise
  return assert_op
def _static_rank_in(actual_rank, given_ranks):
  return actual_rank in given_ranks
def _dynamic_rank_in(actual_rank, given_ranks):
  if len(given_ranks) < 1:
    return ops.convert_to_tensor(False)
  result = math_ops.equal(given_ranks[0], actual_rank)
  for given_rank in given_ranks[1:]:
    result = math_ops.logical_or(
        result, math_ops.equal(given_rank, actual_rank))
  return result
def _assert_ranks_condition(
    x, ranks, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.
  Args:
    x:  Numeric `Tensor`.
    ranks:  Scalar `Tensor`.
    static_condition:   A python function that takes
      `[actual_rank, given_ranks]` and returns `True` if the condition is
      satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_ranks]
      and return `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.
  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  for rank in ranks:
    assert_type(rank, dtypes.int32)
  # Attempt to statically defined rank.
  ranks_static = tuple([tensor_util.constant_value(rank) for rank in ranks])
  if not any(r is None for r in ranks_static):
    for rank_static in ranks_static:
      if rank_static.ndim != 0:
        raise ValueError('Rank must be a scalar.')
    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, ranks_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, ranks_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')
  condition = dynamic_condition(array_ops.rank(x), ranks)
  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  for rank, rank_static in zip(ranks, ranks_static):
    if rank_static is None:
      this_data = ['Rank must be a scalar. Received rank: ', rank]
      rank_check = assert_rank(rank, 0, data=this_data)
      condition = control_flow_ops.with_dependencies([rank_check], condition)
  return control_flow_assert.Assert(condition, data, summarize=summarize)
@tf_export('debugging.assert_rank_in', v1=[])
@dispatch.add_dispatch_support
def assert_rank_in_v2(x, ranks, message=None, name=None):
  """Assert that `x` has a rank in `ranks`.
  This Op checks that the rank of `x` is in `ranks`.
  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.
  Args:
    x: `Tensor`.
    ranks: `Iterable` of scalar `Tensor` objects.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_rank_in".
  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility
  Raises:
    InvalidArgumentError: `x` does not have rank in `ranks`, but the rank cannot
      be statically determined.
    ValueError: If static checks determine `x` has mismatched rank.
  """
  return assert_rank_in(x=x, ranks=ranks, message=message, name=name)
@tf_export(v1=['debugging.assert_rank_in', 'assert_rank_in'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_in')
def assert_rank_in(
    x, ranks, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank in `ranks`.
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):
    output = tf.reduce_sum(x)
  ```
  Args:
    x:  Numeric `Tensor`.
    ranks:  Iterable of scalar `Tensor` objects.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_in".
  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.
  Raises:
    ValueError:  If static checks determine `x` has mismatched rank.
  """
  with ops.name_scope(
      name, 'assert_rank_in', (x,) + tuple(ranks) + tuple(data or [])):
    if not isinstance(x, sparse_tensor.SparseTensor):
      x = ops.convert_to_tensor(x, name='x')
    ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
    message = _message_prefix(message)
    if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
      name = ''
    else:
      name = x.name
    if data is None:
      data = [
          message, 'Tensor %s must have rank in' % name
      ] + list(ranks) + [
          'Received shape: ', array_ops.shape(x)
      ]
    try:
      assert_op = _assert_ranks_condition(x, ranks, _static_rank_in,
                                          _dynamic_rank_in, data, summarize)
    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank in %s.  Received rank %d, '
            'shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise
  return assert_op
@tf_export('debugging.assert_integer', v1=[])
@dispatch.add_dispatch_support
def assert_integer_v2(x, message=None, name=None):
  """Assert that `x` is of integer dtype.
  If `x` has a non-integer type, `message`, as well as the dtype of `x` are
  printed, and `InvalidArgumentError` is raised.
  This can always be checked statically, so this method returns nothing.
  Args:
    x: A `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_integer".
  Raises:
    TypeError:  If `x.dtype` is not a non-quantized integer type.
  """
  assert_integer(x=x, message=message, name=name)
@tf_export(v1=['debugging.assert_integer', 'assert_integer'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_integer')
def assert_integer(x, message=None, name=None):
  """Assert that `x` is of integer dtype.
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.compat.v1.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```
  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_integer".
  Raises:
    TypeError:  If `x.dtype` is anything other than non-quantized integer.
  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  with ops.name_scope(name, 'assert_integer', [x]):
    x = ops.convert_to_tensor(x, name='x')
    if not x.dtype.is_integer:
      if context.executing_eagerly():
        name = 'tensor'
      else:
        name = x.name
      err_msg = (
          '%sExpected "x" to be integer type.  Found: %s of dtype %s'
          % (_message_prefix(message), name, x.dtype))
      raise TypeError(err_msg)
    return control_flow_ops.no_op('statically_determined_was_integer')
@tf_export('debugging.assert_type', v1=[])
@dispatch.add_dispatch_support
def assert_type_v2(tensor, tf_type, message=None, name=None):
  """Asserts that the given `Tensor` is of the specified type.
  This can always be checked statically, so this method returns nothing.
  Example:
  >>> a = tf.Variable(1.0)
  >>> tf.debugging.assert_type(a, tf_type= tf.float32)
  >>> b = tf.constant(21)
  >>> tf.debugging.assert_type(b, tf_type=tf.bool)
  Traceback (most recent call last):
  ...
  TypeError: ...
  >>> c = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2],
  ...  dense_shape=[3, 4])
  >>> tf.debugging.assert_type(c, tf_type= tf.int32)
  Args:
    tensor: A `Tensor`, `SparseTensor` or `tf.Variable` .
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name for this operation. Defaults to "assert_type"
  Raises:
    TypeError: If the tensor's data type doesn't match `tf_type`.
  """
  assert_type(tensor=tensor, tf_type=tf_type, message=message, name=name)
@tf_export(v1=['debugging.assert_type', 'assert_type'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_type')
def assert_type(tensor, tf_type, message=None, name=None):
  """Statically asserts that the given `Tensor` is of the specified type.
  Args:
    tensor: A `Tensor` or `SparseTensor`.
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name to give this `Op`.  Defaults to "assert_type"
  Raises:
    TypeError: If the tensors data type doesn't match `tf_type`.
  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  tf_type = dtypes.as_dtype(tf_type)
  with ops.name_scope(name, 'assert_type', [tensor]):
    if not isinstance(tensor, sparse_tensor.SparseTensor):
      tensor = ops.convert_to_tensor(tensor, name='tensor')
    if tensor.dtype != tf_type:
      raise TypeError(
          f'{_message_prefix(message)}{getattr(tensor, "name", "tensor")}'
          f' must be of type {tf_type!r}; got {tensor.dtype!r}')
    return control_flow_ops.no_op('statically_determined_correct_type')
def _dimension_sizes(x):
  """Gets the dimension sizes of a tensor `x`.
  If a size can be determined statically it is returned as an integer,
  otherwise as a tensor.
  If `x` is a scalar it is treated as rank 1 size 1.
  Args:
    x: A `Tensor`.
  Returns:
    Dimension sizes.
  """
  dynamic_shape = array_ops.shape(x)
  rank = x.get_shape().rank
  rank_is_known = rank is not None
  if rank_is_known and rank == 0:
    return (1,)
  if rank_is_known and rank > 0:
    static_shape = x.get_shape().as_list()
    sizes = [
        int(size) if size is not None else dynamic_shape[i]
        for i, size in enumerate(static_shape)
    ]
    return sizes
  has_rank_zero = math_ops.equal(array_ops.rank(x), 0)
  return cond.cond(
      has_rank_zero, lambda: array_ops.constant([1]), lambda: dynamic_shape)
def _symbolic_dimension_sizes(symbolic_shape):
  # If len(symbolic_shape) == 0 construct a tuple
  if not symbolic_shape:
    return tuple([1])
  return symbolic_shape
def _has_known_value(dimension_size):
  not_none = dimension_size is not None
  try:
    int(dimension_size)
    can_be_parsed_as_int = True
  except (ValueError, TypeError):
    can_be_parsed_as_int = False
  return not_none and can_be_parsed_as_int
def _is_symbol_for_any_size(symbol):
  return symbol in [None, '.']
_TensorDimSizes = collections.namedtuple(
    '_TensorDimSizes',
    ['x', 'unspecified_dim', 'actual_sizes', 'symbolic_sizes'])
@tf_export('debugging.assert_shapes', v1=[])
@dispatch.add_dispatch_support
def assert_shapes_v2(shapes, data=None, summarize=None, message=None,
                     name=None):
  """Assert tensor shapes and dimension size relationships between tensors.
  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.
  Example:
  >>> n = 10
  >>> q = 3
  >>> d = 7
  >>> x = tf.zeros([n,q])
  >>> y = tf.ones([n,d])
  >>> param = tf.Variable([1.0, 2.0, 3.0])
  >>> scalar = 1.0
  >>> tf.debugging.assert_shapes([
  ...  (x, ('N', 'Q')),
  ...  (y, ('N', 'D')),
  ...  (param, ('Q',)),
  ...  (scalar, ()),
  ... ])
  >>> tf.debugging.assert_shapes([
  ...   (x, ('N', 'D')),
  ...   (y, ('N', 'D'))
  ... ])
  Traceback (most recent call last):
  ...
  ValueError: ...
  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.
  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.
  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.
  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.
  Args:
    shapes: dictionary with (`Tensor` to shape) items, or a list of
      (`Tensor`, shape) tuples. A shape must be an iterable.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_shapes".
  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  """
  assert_shapes(
      shapes, data=data, summarize=summarize, message=message, name=name)
@tf_export(v1=['debugging.assert_shapes'])
@dispatch.add_dispatch_support
def assert_shapes(shapes, data=None, summarize=None, message=None, name=None):
  """Assert tensor shapes and dimension size relationships between tensors.
  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.
  Example:
  >>> n = 10
  >>> q = 3
  >>> d = 7
  >>> x = tf.zeros([n,q])
  >>> y = tf.ones([n,d])
  >>> param = tf.Variable([1.0, 2.0, 3.0])
  >>> scalar = 1.0
  >>> tf.debugging.assert_shapes([
  ...  (x, ('N', 'Q')),
  ...  (y, ('N', 'D')),
  ...  (param, ('Q',)),
  ...  (scalar, ()),
  ... ])
  >>> tf.debugging.assert_shapes([
  ...   (x, ('N', 'D')),
  ...   (y, ('N', 'D'))
  ... ])
  Traceback (most recent call last):
  ...
  ValueError: ...
  Example of adding a dependency to an operation:
  ```python
  with tf.control_dependencies([tf.assert_shapes(shapes)]):
    output = tf.matmul(x, y, transpose_a=True)
  ```
  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.
  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.
  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.
  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.
  Args:
    shapes: A list of (`Tensor`, `shape`) tuples, wherein `shape` is the
      expected shape of `Tensor`. See the example code above. The `shape` must
      be an iterable. Each element of the iterable can be either a concrete
      integer value or a string that abstractly represents the dimension.
      For example,
        - `('N', 'Q')` specifies a 2D shape wherein the first and second
          dimensions of shape may or may not be equal.
        - `('N', 'N', 'Q')` specifies a 3D shape wherein the first and second
          dimensions are equal.
        - `(1, 'N')` specifies a 2D shape wherein the first dimension is
          exactly 1 and the second dimension can be any value.
      Note that the abstract dimension letters take effect across different
      tuple elements of the list. For example,
      `tf.debugging.assert_shapes([(x, ('N', 'A')), (y, ('N', 'B'))]` asserts
      that both `x` and `y` are rank-2 tensors and their first dimensions are
      equal (`N`).
      `shape` can also be a `tf.TensorShape`.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_shapes".
  Returns:
    Op raising `InvalidArgumentError` unless all shape constraints are
    satisfied.
    If static checks determine all constraints are satisfied, a `no_op` is
    returned.
  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  """
  # If the user manages to assemble a dict containing tensors (possible in
  # Graph mode only), make sure we still accept that.
  if isinstance(shapes, dict):
    shapes = shapes.items()
  message_prefix = _message_prefix(message)
  with ops.name_scope(name, 'assert_shapes', [shapes, data]):
    # Shape specified as None implies no constraint
    shape_constraints = [(x if isinstance(x, sparse_tensor.SparseTensor) else
                          ops.convert_to_tensor(x), s)
                         for x, s in shapes if s is not None]
    executing_eagerly = context.executing_eagerly()
    def tensor_name(x):
      if executing_eagerly or isinstance(x, sparse_tensor.SparseTensor):
        return _shape_and_dtype_str(x)
      return x.name
    tensor_dim_sizes = []
    for tensor, symbolic_shape in shape_constraints:
      is_iterable = (
          hasattr(symbolic_shape, '__iter__') or
          hasattr(symbolic_shape, '__getitem__')  # For Python 2 compat.
      )
      if not is_iterable:
        raise ValueError(
            '%s'
            'Tensor %s.  Specified shape must be an iterable.  '
            'An iterable has the attribute `__iter__` or `__getitem__`.  '
            'Received specified shape: %s' %
            (message_prefix, tensor_name(tensor), symbolic_shape))
      # We convert this into a tuple to handle strings, lists and numpy arrays
      symbolic_shape_tuple = tuple(symbolic_shape)
      tensors_specified_innermost = False
      for i, symbol in enumerate(symbolic_shape_tuple):
        if symbol not in [Ellipsis, '*']:
          continue
        if i != 0:
          raise ValueError(
              '%s'
              'Tensor %s specified shape index %d.  '
              'Symbol `...` or `*` for a variable number of '
              'unspecified dimensions is only allowed as the first entry' %
              (message_prefix, tensor_name(tensor), i))
        tensors_specified_innermost = True
      # Only include the size of the specified dimensions since the 0th symbol
      # is either ellipsis or *
      tensor_dim_sizes.append(
          _TensorDimSizes(
              tensor, tensors_specified_innermost, _dimension_sizes(tensor),
              _symbolic_dimension_sizes(
                  symbolic_shape_tuple[1:]
                  if tensors_specified_innermost else symbolic_shape_tuple)))
    rank_assertions = []
    for sizes in tensor_dim_sizes:
      rank = len(sizes.symbolic_sizes)
      rank_zero_or_one = rank in [0, 1]
      if sizes.unspecified_dim:
        if rank_zero_or_one:
          # No assertion of rank needed as `x` only need to have rank at least
          # 0. See elif rank_zero_or_one case comment.
          continue
        assertion = assert_rank_at_least(
            x=sizes.x,
            rank=rank,
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      elif rank_zero_or_one:
        # Rank 0 is treated as rank 1 size 1, i.e. there is
        # no distinction between the two in terms of rank.
        # See _dimension_sizes.
        assertion = assert_rank_in(
            x=sizes.x,
            ranks=[0, 1],
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      else:
        assertion = assert_rank(
            x=sizes.x,
            rank=rank,
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      rank_assertions.append(assertion)
    size_assertions = []
    size_specifications = {}
    for sizes in tensor_dim_sizes:
      for i, size_symbol in enumerate(sizes.symbolic_sizes):
        if _is_symbol_for_any_size(size_symbol):
          # Size specified as any implies no constraint
          continue
        if sizes.unspecified_dim:
          tensor_dim = i - len(sizes.symbolic_sizes)
        else:
          tensor_dim = i
        if size_symbol in size_specifications or _has_known_value(size_symbol):
          if _has_known_value(size_symbol):
            specified_size = int(size_symbol)
            size_check_message = 'Specified explicitly'
          else:
            specified_size, specified_by_y, specified_at_dim = (
                size_specifications[size_symbol])
            size_check_message = (
                'Specified by tensor %s dimension %d' %
                (tensor_name(specified_by_y), specified_at_dim))
          # This is extremely subtle. If actual_sizes is dynamic, we must
          # make sure a control dependency is inserted here so that this slice
          # can not execute until the rank is asserted to be enough for the
          # slice to not fail.
          with ops.control_dependencies(rank_assertions):
            actual_size = sizes.actual_sizes[tensor_dim]
          if _has_known_value(actual_size) and _has_known_value(specified_size):
            if int(actual_size) != int(specified_size):
              raise ValueError(
                  '%s%s.  Tensor %s dimension %s must have size %d.  '
                  'Received size %d, shape %s' %
                  (message_prefix, size_check_message, tensor_name(sizes.x),
                   tensor_dim, specified_size, actual_size,
                   sizes.x.get_shape()))
            # No dynamic assertion needed
            continue
          condition = math_ops.equal(
              ops.convert_to_tensor(actual_size),
              ops.convert_to_tensor(specified_size))
          data_ = data
          if data is None:
            data_ = [
                message_prefix, size_check_message,
                'Tensor %s dimension' % tensor_name(sizes.x), tensor_dim,
                'must have size', specified_size, 'Received shape: ',
                array_ops.shape(sizes.x)
            ]
          size_assertions.append(
              control_flow_assert.Assert(condition, data_, summarize=summarize))
        else:
          # Not sure if actual_sizes is a constant, but for safety, guard
          # on rank. See explanation above about actual_sizes need for safety.
          with ops.control_dependencies(rank_assertions):
            size = sizes.actual_sizes[tensor_dim]
          size_specifications[size_symbol] = (size, sizes.x, tensor_dim)
  # Ensure both assertions actually occur.
  with ops.control_dependencies(rank_assertions):
    shapes_assertion = control_flow_ops.group(size_assertions)
  return shapes_assertion
# pylint: disable=line-too-long
def _get_diff_for_monotonic_comparison(x):
  """Gets the difference x[1:] - x[:-1]."""
  x = array_ops.reshape(x, [-1])
  if not is_numeric_tensor(x):
    raise TypeError('Expected x to be numeric, instead found: %s' % x)
  # If x has less than 2 elements, there is nothing to compare.  So return [].
  is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
  short_result = lambda: ops.convert_to_tensor([], dtype=x.dtype)
  # With 2 or more elements, return x[1:] - x[:-1]
  s_len = array_ops.shape(x) - 1
  diff = lambda: array_ops.strided_slice(x, [1], [1] + s_len)- array_ops.strided_slice(x, [0], s_len)
  return cond.cond(is_shorter_than_two, short_result, diff)
