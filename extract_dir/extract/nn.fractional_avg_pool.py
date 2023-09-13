@tf_export("nn.fractional_avg_pool", v1=[])
@dispatch.add_dispatch_support
def fractional_avg_pool_v2(value,
                           pooling_ratio,
                           pseudo_random=False,
                           overlapping=False,
                           seed=0,
                           name=None):  # pylint: disable=redefined-builtin
  r"""Performs fractional average pooling on the input.
  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.
  Args:
    value: A `Tensor`. 4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length >= 4.  Pooling ratio for
      each dimension of `value`, currently only supports row and col dimension
      and should be >= 1.0. For example, a valid pooling ratio looks like [1.0,
      1.44, 1.73, 1.0]. The first and last elements must be 1.0 because we don't
      allow pooling on batch and channels dimensions.  1.44 and 1.73 are pooling
      ratio on height and width dimensions respectively.
    pseudo_random: An optional `bool`.  Defaults to `False`. When set to `True`,
      generates the pooling sequence in a pseudorandom fashion, otherwise, in a
      random fashion. Check paper (Graham, 2015) for difference between
      pseudorandom and random.
    overlapping: An optional `bool`.  Defaults to `False`.  When set to `True`,
      it means when pooling, the values at the boundary of adjacent pooling
      cells are used by both cells. For example:
      `index  0  1  2  3  4`
      `value  20 5  16 3  7`
      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
      twice.  The result would be [20, 16] for fractional avg pooling.
    seed: An optional `int`.  Defaults to `0`.  If set to be non-zero, the
      random number generator is seeded by the given seed.  Otherwise it is
      seeded by a random seed.
    name: A name for the operation (optional).
  Returns:
  A tuple of `Tensor` objects (`output`, `row_pooling_sequence`,
  `col_pooling_sequence`).
    output: Output `Tensor` after fractional avg pooling.  Has the same type as
      `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  if seed == 0:
    return gen_nn_ops.fractional_avg_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=False,
                                          seed=0, seed2=0, name=name)
  else:
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_nn_ops.fractional_avg_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=True,
                                          seed=seed1, seed2=seed2, name=name)
