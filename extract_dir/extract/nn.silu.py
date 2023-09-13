@tf_export("nn.silu", "nn.swish")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def swish(features, beta=1.0):
  # pylint: disable=g-doc-args
  """Computes the SiLU or Swish activation function: `x * sigmoid(beta * x)`.
  beta : Hyperparameter for Swish activation function. Default value 1.0.
  The SiLU activation function was introduced in "Gaussian Error Linear Units
  (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415) and
  "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
  Reinforcement Learning"
  [Elfwing et al. 2017](https://arxiv.org/abs/1702.03118) and was independently
  discovered (and called swish) in "Searching for Activation Functions"
  [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)
  Args:
    features: A `Tensor` representing preactivation values.
    beta: A 'Tensor' representing value of beta hyperparameter.
  Returns:
    The activation value.
  """
  # pylint: enable=g-doc-args
  features = ops.convert_to_tensor(features, name="features")
  beta = ops.convert_to_tensor(beta, name="beta")
  beta = math_ops.cast(beta, features.dtype)
  @custom_gradient.custom_gradient
  def swish_impl(features, beta):
    def grad(dy):
      """Gradient for the Swish activation function."""
      # Naively, x * tf.nn.sigmoid(x) requires keeping both x and sigmoid(x)
      # around for backprop, effectively doubling the tensor's memory
      # consumption. We use a control dependency here so that sigmoid(features)
      # is re-computed during backprop (the control dep prevents it being
      # de-duped with the forward pass) and we can free the sigmoid(features)
      # expression immediately after use during the forward pass.
      with ops.control_dependencies([dy]):
        sigmoid_features = math_ops.sigmoid(beta * features)
      activation_grad = (
          sigmoid_features * (1.0 + (beta * features) *
                              (1.0 - sigmoid_features)))
      beta_grad = math_ops.reduce_sum(
          dy * math_ops.square(features) * sigmoid_features *
          (1.0 - sigmoid_features))
      return (dy * activation_grad, beta_grad)
    return features * math_ops.sigmoid(beta * features), grad
  return swish_impl(features, beta)
