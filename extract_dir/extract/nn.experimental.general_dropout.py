@tf_export("nn.experimental.general_dropout")
@dispatch.add_dispatch_support
def general_dropout(x, rate, uniform_sampler, noise_shape=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.
  Please see `tf.nn.experimental.stateless_dropout` for an overview
  of dropout.
  Unlike `tf.nn.experimental.stateless_dropout`, here you can supply a
  custom sampler function `uniform_sampler` that (given a shape and a
  dtype) generates a random, `Uniform[0, 1)`-distributed tensor (of
  that shape and dtype).  `uniform_sampler` can be
  e.g. `tf.random.stateless_random_uniform` or
  `tf.random.Generator.uniform`.
  For example, if you are using `tf.random.Generator` to generate
  random numbers, you can use this code to do dropouts:
  >>> g = tf.random.Generator.from_seed(7)
  >>> sampler = g.uniform
  >>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
  >>> rate = 0.5
  >>> tf.nn.experimental.general_dropout(x, rate, sampler)
  <tf.Tensor: shape=(5,), ..., numpy=array([ 0. ,  4.4,  6.6,  8.8, 11. ], ...)>
  >>> tf.nn.experimental.general_dropout(x, rate, sampler)
  <tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 0. , 8.8, 0. ], ...)>
  It has better performance than using
  `tf.nn.experimental.stateless_dropout` and
  `tf.random.Generator.make_seeds`:
  >>> g = tf.random.Generator.from_seed(7)
  >>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
  >>> rate = 0.5
  >>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
  <tf.Tensor: shape=(5,), ..., numpy=array([ 2.2,  4.4,  6.6,  0. , 11. ], ...)>
  >>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
  <tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 6.6, 8.8, 0. ], ...>
  because generating and consuming seeds cost extra
  computation. `tf.nn.experimental.general_dropout` can let you avoid
  them.
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    uniform_sampler: a callable of signature `(shape, dtype) ->
      Tensor[shape, dtype]`, used to generate a tensor of uniformly-distributed
      random numbers in the range `[0, 1)`, of the given shape and dtype.
    noise_shape: A 1-D integer `Tensor`, representing the
      shape for randomly generated keep/drop flags.
    name: A name for this operation.
  Returns:
    A Tensor of the same shape and dtype of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  def dummy_rng_step():
    pass
  return _dropout(x=x, rate=rate, noise_shape=noise_shape,
                  uniform_sampler=uniform_sampler,
                  dummy_rng_step=dummy_rng_step, name=name,
                  default_name="general_dropout")
