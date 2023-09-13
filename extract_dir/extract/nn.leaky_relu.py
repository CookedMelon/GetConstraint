@tf_export("nn.leaky_relu")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  Source: [Rectifier Nonlinearities Improve Neural Network Acoustic Models.
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013]
  (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  References:
    Rectifier Nonlinearities Improve Neural Network Acoustic Models:
      [Maas et al., 2013]
      (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.693.1422)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.1422&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]) as name:
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.cast(features, dtypes.float32)
    if isinstance(alpha, np.ndarray):
      alpha = alpha.item()
    return gen_nn_ops.leaky_relu(features, alpha=alpha, name=name)
