@tf_export("nn.isotonic_regression", v1=[])
@dispatch.add_dispatch_support
def isotonic_regression(inputs, decreasing=True, axis=-1):
  r"""Solves isotonic regression problems along the given axis.
  For each vector x, the problem solved is
  $$\argmin_{y_1 >= y_2 >= ... >= y_n} \sum_i (x_i - y_i)^2.$$
  As the solution is component-wise constant, a second tensor is returned that
  encodes the segments. The problems are solved over the given axis.
  Consider the following example, where we solve a batch of two problems. The
  first input is [3, 1, 2], while the second [1, 3, 4] (as the axis is 1).
  >>> x = tf.constant([[3, 1, 2], [1, 3, 4]], dtype=tf.float32)
  >>> y, segments = tf.nn.isotonic_regression(x, axis=1)
  >>> y  # The solution.
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[3.       , 1.5      , 1.5      ],
         [2.6666667, 2.6666667, 2.6666667]], dtype=float32)>
  Note that the first solution has two blocks [2] and [1.5, 1.5]. The second
  solution is constant, and thus has a single segment. These segments are
  exactly what the second returned tensor encodes:
  >>> segments
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[0, 1, 1],
         [0, 0, 0]], dtype=int32)>
  Args:
    inputs: A tensor holding the inputs.
    decreasing: If set to False, the inequalities in the optimizing constrained
      are flipped.
    axis: The axis along which the problems should be solved.
  Returns:
    output: The solutions, same shape as type as the input.
    segments: An int32 tensor, same shape as the input indicating the segments
      that have the same value. Specifically, those positions that have the same
      value correspond to the same segment. These values start at zero, and are
      monotonously increasing for each solution.
  """
  type_promotions = {
      # Float types get mapped to themselves, int8/16 to float32, rest to double
      dtypes.float32:
          dtypes.float32,
      dtypes.half:
          dtypes.half,
      dtypes.bfloat16:
          dtypes.bfloat16,
      dtypes.int8:
          dtypes.float32,
      dtypes.int16:
          dtypes.float32,
  }
  inputs = ops.convert_to_tensor(inputs)
  try:
    output_dtype = type_promotions[inputs.dtype]
  except KeyError:
    output_dtype = dtypes.float64
  def compute_on_matrix(matrix, name=None):
    iso_fn = functools.partial(
        gen_nn_ops.isotonic_regression, output_dtype=output_dtype, name=name)
    if decreasing:
      return iso_fn(matrix)
    else:
      output, segments = iso_fn(-matrix)
      return -output, segments
  return _wrap_2d_function(inputs, compute_on_matrix, axis)
