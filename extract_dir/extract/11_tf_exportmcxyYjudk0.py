"/home/cc/Workspace/tfconstraint/python/ops/check_ops.py"
@tf_export(
    'math.is_non_decreasing',
    v1=[
        'math.is_non_decreasing', 'debugging.is_non_decreasing',
        'is_non_decreasing'
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_non_decreasing',
                                  'is_non_decreasing')
def is_non_decreasing(x, name=None):
  """Returns `True` if `x` is non-decreasing.
  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
  If `x` has less than two elements, it is trivially non-decreasing.
  See also:  `is_strictly_increasing`
  >>> x1 = tf.constant([1.0, 1.0, 3.0])
  >>> tf.math.is_non_decreasing(x1)
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> x2 = tf.constant([3.0, 1.0, 2.0])
  >>> tf.math.is_non_decreasing(x2)
  <tf.Tensor: shape=(), dtype=bool, numpy=False>
  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).  Defaults to "is_non_decreasing"
  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.
  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.name_scope(name, 'is_non_decreasing', [x]):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less_equal = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less_equal(zero, diff))
