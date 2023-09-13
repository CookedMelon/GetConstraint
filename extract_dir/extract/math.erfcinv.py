@tf_export("math.erfcinv")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def erfcinv(x, name=None):
  """Computes the inverse of complementary error function.
  Given `x`, compute the inverse complementary error function of `x`.
  This function is the inverse of `tf.math.erfc`, and is defined on
  `[0, 2]`.
  >>> tf.math.erfcinv([0., 0.2, 1., 1.5, 2.])
  <tf.Tensor: shape=(5,), dtype=float32, numpy=
  array([       inf,  0.9061935, -0.       , -0.4769363,       -inf],
        dtype=float32)>
  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse complementary error function of `x`.
  @compatibility(numpy)
  Equivalent to scipy.special.erfcinv
  @end_compatibility
  """
  with ops.name_scope(name, "erfcinv", [x]):
    x = ops.convert_to_tensor(x, name="start")
    return -ndtri(0.5 * x) * np.sqrt(0.5)
