@tf_export("nn.dropout", v1=[])
@dispatch.add_dispatch_support
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.
  Warning: You should consider using
  `tf.nn.experimental.stateless_dropout` instead of this function. The
  difference between `tf.nn.experimental.stateless_dropout` and this
  function is analogous to the difference between
  `tf.random.stateless_uniform` and `tf.random.uniform`. Please see
  [Random number
  generation](https://www.tensorflow.org/guide/random_numbers) guide
  for a detailed description of the various RNG systems in TF. As the
  guide states, legacy stateful RNG ops like `tf.random.uniform` and
  `tf.nn.dropout` are not deprecated yet but highly discouraged,
  because their states are hard to control.
  Note: The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.
  See also: `tf.keras.layers.Dropout` for a dropout layer.
  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.
  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
  expected value is preserved.
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.5, seed = 1).numpy()
  array([[2., 0., 0., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 0., 2., 0., 2.]], dtype=float32)
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.8, seed = 1).numpy()
  array([[0., 0., 0., 5., 5.],
       [0., 5., 0., 5., 0.],
       [5., 0., 5., 0., 5.]], dtype=float32)
  >>> tf.nn.dropout(x, rate = 0.0) == x
  <tf.Tensor: shape=(3, 5), dtype=bool, numpy=
    array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])>
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions. This is useful for dropping whole
  channels from an image or sequence. For example:
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,10])
  >>> tf.nn.dropout(x, rate = 2/3, noise_shape=[1,10], seed=1).numpy()
  array([[0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.]], dtype=float32)
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D integer `Tensor`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  uniform_sampler = functools.partial(random_ops.random_uniform, seed=seed)
  def dummy_rng_step():
    random_seed.get_seed(seed)
  return _dropout(x=x, rate=rate, noise_shape=noise_shape,
                  uniform_sampler=uniform_sampler,
                  dummy_rng_step=dummy_rng_step, name=name,
                  default_name="dropout")
