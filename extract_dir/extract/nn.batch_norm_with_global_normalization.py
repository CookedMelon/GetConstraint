@tf_export("nn.batch_norm_with_global_normalization", v1=[])
@dispatch.add_dispatch_support
def batch_norm_with_global_normalization_v2(input,
                                            mean,
                                            variance,
                                            beta,
                                            gamma,
                                            variance_epsilon,
                                            scale_after_normalization,
                                            name=None):
  """Batch normalization.
  This op is deprecated. See `tf.nn.batch_normalization`.
  Args:
    input: A 4D input Tensor.
    mean: A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    variance: A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    scale_after_normalization: A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for this operation (optional).
  Returns:
     A batch-normalized `t`.
  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift:
      [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
  return batch_norm_with_global_normalization(t=input,
                                              m=mean,
                                              v=variance,
                                              beta=beta,
                                              gamma=gamma,
                                              variance_epsilon=variance_epsilon,
                                              scale_after_normalization=scale_after_normalization,
                                              name=name)
