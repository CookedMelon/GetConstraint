@tf_export(v1=["nn.crelu"])
@dispatch.add_dispatch_support
def crelu(features, name=None, axis=-1):
  """Computes Concatenated ReLU.
  Concatenates a ReLU which selects only the positive part of the activation
  with a ReLU which selects only the *negative* part of the activation.
  Note that as a result this non-linearity doubles the depth of the activations.
  Source: [Understanding and Improving Convolutional Neural Networks via
  Concatenated Rectified Linear Units. W. Shang, et
  al.](https://arxiv.org/abs/1603.05201)
  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).
    axis: The axis that the output values are concatenated along. Default is -1.
  Returns:
    A `Tensor` with the same type as `features`.
  References:
    Understanding and Improving Convolutional Neural Networks via Concatenated
    Rectified Linear Units:
      [Shang et al., 2016](http://proceedings.mlr.press/v48/shang16)
      ([pdf](http://proceedings.mlr.press/v48/shang16.pdf))
  """
  with ops.name_scope(name, "CRelu", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    c = array_ops.concat([features, -features], axis, name=name)  # pylint: disable=invalid-unary-operand-type
    return gen_nn_ops.relu(c)
