"/home/cc/Workspace/tfconstraint/python/ops/check_ops.py"
@tf_export(
    'math.is_strictly_increasing',
    v1=[
        'math.is_strictly_increasing', 'debugging.is_strictly_increasing',
        'is_strictly_increasing'
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_strictly_increasing',
                                  'is_strictly_increasing')
def is_strictly_increasing(x, name=None):
  """Returns `True` if `x` is strictly increasing.
  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
  If `x` has less than two elements, it is trivially strictly increasing.
  See also:  `is_non_decreasing`
  >>> x1 = tf.constant([1.0, 2.0, 3.0])
  >>> tf.math.is_strictly_increasing(x1)
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> x2 = tf.constant([3.0, 1.0, 2.0])
  >>> tf.math.is_strictly_increasing(x2)
  <tf.Tensor: shape=(), dtype=bool, numpy=False>
  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).
      Defaults to "is_strictly_increasing"
  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.
  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.name_scope(name, 'is_strictly_increasing', [x]):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less(zero, diff))
def _assert_same_base_type(items, expected_type=None):
  r"""Asserts all items are of the same base type.
  Args:
    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
        `Operation`, or `IndexedSlices`). Can include `None` elements, which
        will be ignored.
    expected_type: Expected type. If not specified, assert all items are
        of the same base type.
  Returns:
    Validated type, or none if neither expected_type nor items provided.
  Raises:
    ValueError: If any types do not match.
  """
  original_expected_type = expected_type
  mismatch = False
  for item in items:
    if item is not None:
      item_type = item.dtype.base_dtype
      if not expected_type:
        expected_type = item_type
      elif expected_type != item_type:
        mismatch = True
        break
  if mismatch:
    # Loop back through and build up an informative error message (this is very
    # slow, so we don't do it unless we found an error above).
    expected_type = original_expected_type
    original_item_str = None
    for item in items:
      if item is not None:
        item_type = item.dtype.base_dtype
        if not expected_type:
          expected_type = item_type
          original_item_str = item.name if hasattr(item, 'name') else str(item)
        elif expected_type != item_type:
          raise ValueError('%s, type=%s, must be of the same type (%s)%s.' % (
              item.name if hasattr(item, 'name') else str(item),
              item_type, expected_type,
              (' as %s' % original_item_str) if original_item_str else ''))
    return expected_type  # Should be unreachable
  else:
    return expected_type
