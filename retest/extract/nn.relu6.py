@tf_export("nn.relu6")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def relu6(features, name=None):
  """Computes Rectified Linear 6: `min(max(features, 0), 6)`.
  In comparison with `tf.nn.relu`, relu6 activation functions have shown to
  empirically perform better under low-precision conditions (e.g. fixed point
  inference) by encouraging the model to learn sparse features earlier.
  Source: [Convolutional Deep Belief Networks on CIFAR-10: Krizhevsky et al.,
  2010](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf).
  For example:
  >>> x = tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float32)
  >>> y = tf.nn.relu6(x)
  >>> y.numpy()
  array([0., 0., 0., 6., 6.], dtype=float32)
  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same type as `features`.
  References:
    Convolutional Deep Belief Networks on CIFAR-10:
      Krizhevsky et al., 2010
      ([pdf](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf))
  """
  with ops.name_scope(name, "Relu6", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    return gen_nn_ops.relu6(features, name=name)
