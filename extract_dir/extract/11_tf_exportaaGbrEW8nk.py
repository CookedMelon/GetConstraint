"/home/cc/Workspace/tfconstraint/python/ops/candidate_sampling_ops.py"
@tf_export(
    'random.log_uniform_candidate_sampler',
    v1=[
        'random.log_uniform_candidate_sampler',
        'nn.log_uniform_candidate_sampler'
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('nn.log_uniform_candidate_sampler')
def log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique,
                                  range_max, seed=None, name=None):
  """Samples a set of classes using a log-uniform (Zipfian) base distribution.
  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max)`.
  See the [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf)
  for a quick course on Candidate Sampling.
  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.
  The base distribution for this operation is an approximately log-uniform
  or Zipfian distribution:
  `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`
  This sampler is useful when the target classes approximately follow such
  a distribution - for example, if the classes represent words in a lexicon
  sorted in decreasing order of frequency. If your classes are not ordered by
  decreasing frequency, do not use this op.
  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in the [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.
  Note that this function (and also other `*_candidate_sampler`
  functions) only gives you the ingredients to implement the various
  Candidate Sampling algorithms listed in the big table in the
  [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf). You
  still need to implement the algorithms yourself.
  For example, according to that table, the phrase "negative samples"
  may mean different things in different algorithms. For instance, in
  NCE, "negative samples" means `S_i` (which is just the sampled
  classes) which may overlap with true classes, while in Sampled
  Logistic, "negative samples" means `S_i - T_i` which excludes the
  true classes. The return value `sampled_candidates` corresponds to
  `S_i`, not to any specific definition of "negative samples" in any
  specific algorithm. It's your responsibility to pick an algorithm
  and calculate the "negative samples" defined by that algorithm
  (e.g. `S_i - T_i`).
  As another example, the `true_classes` argument is for calculating
  the `true_expected_count` output (as a by-product of this function's
  main calculation), which may be needed by some algorithms (according
  to that table). It's not for excluding true classes in the return
  value `sampled_candidates`. Again that step is algorithm-specific
  and should be carried out by you.
  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).
  Returns:
    sampled_candidates: A tensor of type `int64` and shape
      `[num_sampled]`. The sampled classes. As noted above,
      `sampled_candidates` may overlap with true classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_candidate_sampling_ops.log_uniform_candidate_sampler(
      true_classes, num_true, num_sampled, unique, range_max, seed=seed1,
      seed2=seed2, name=name)
