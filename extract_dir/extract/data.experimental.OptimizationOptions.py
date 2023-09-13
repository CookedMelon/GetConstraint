@tf_export("data.experimental.OptimizationOptions")
class OptimizationOptions(options_lib.OptionsBase):
  """Represents options for dataset optimizations.
  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.
  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.apply_default_optimizations = False
  dataset = dataset.with_options(options)
  ```
  """
  apply_default_optimizations = options_lib.create_option(
      name="apply_default_optimizations",
      ty=bool,
      docstring=
      "Whether to apply default graph optimizations. If False, only graph "
      "optimizations that have been explicitly enabled will be applied.")
  filter_fusion = options_lib.create_option(
      name="filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse filter transformations. If None, defaults to False.")
  filter_parallelization = options_lib.create_option(
      name="filter_parallelization",
      ty=bool,
      docstring=
      "Whether to parallelize stateless filter transformations. If None, "
      "defaults to False.")
  inject_prefetch = options_lib.create_option(
      name="inject_prefetch",
      ty=bool,
      docstring=
      "Whether to inject prefetch transformation as the last transformation "
      "when the last transformation is a synchronous transformation. If None, "
      "defaults to True.")
  map_and_batch_fusion = options_lib.create_option(
      name="map_and_batch_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and batch transformations. If None, defaults to "
      "True.")
  map_and_filter_fusion = options_lib.create_option(
      name="map_and_filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and filter transformations. If None, defaults to "
      "False.")
  map_fusion = options_lib.create_option(
      name="map_fusion",
      ty=bool,
      docstring="Whether to fuse map transformations. If None, defaults to "
      "False.")
  map_parallelization = options_lib.create_option(
      name="map_parallelization",
      ty=bool,
      docstring=
      "Whether to parallelize stateless map transformations. If None, defaults "
      "to True.")
  noop_elimination = options_lib.create_option(
      name="noop_elimination",
      ty=bool,
      docstring=
      "Whether to eliminate no-op transformations. If None, defaults to True.")
  parallel_batch = options_lib.create_option(
      name="parallel_batch",
      ty=bool,
      docstring="Whether to parallelize copying of batch elements. If None, "
      "defaults to True.")
  shuffle_and_repeat_fusion = options_lib.create_option(
      name="shuffle_and_repeat_fusion",
      ty=bool,
      docstring="Whether to fuse shuffle and repeat transformations. If None, "
      "defaults to True.")
  warm_start = options_lib.create_option(
      name="warm_start",
      ty=bool,
      docstring=(
          "Whether to start background threads of asynchronous transformations"
          " upon iterator creation (as opposed to upon first call to"
          " `GetNext`). If None, defaults to False.  It should be noted that"
          " this possibly improves the latency of the initial 'GetNext' call at"
          " the expense of requiring more memory to hold prefetched elements"
          " between the time of iterator construction and usage."
      ),
      default_factory=lambda: True if test_mode.TEST_MODE else False,
  )
  def _to_proto(self):
    pb = dataset_options_pb2.OptimizationOptions()
    if self.apply_default_optimizations is not None:
      pb.apply_default_optimizations = self.apply_default_optimizations
    if self.filter_fusion is not None:
      pb.filter_fusion = self.filter_fusion
    if self.filter_parallelization is not None:
      pb.filter_parallelization = self.filter_parallelization
    if self.inject_prefetch is not None:
      pb.inject_prefetch = self.inject_prefetch
    if self.map_and_batch_fusion is not None:
      pb.map_and_batch_fusion = self.map_and_batch_fusion
    if self.map_and_filter_fusion is not None:
      pb.map_and_filter_fusion = self.map_and_filter_fusion
    if self.map_fusion is not None:
      pb.map_fusion = self.map_fusion
    if self.map_parallelization is not None:
      pb.map_parallelization = self.map_parallelization
    if self.noop_elimination is not None:
      pb.noop_elimination = self.noop_elimination
    if self.parallel_batch is not None:
      pb.parallel_batch = self.parallel_batch
    if self.shuffle_and_repeat_fusion is not None:
      pb.shuffle_and_repeat_fusion = self.shuffle_and_repeat_fusion
    if self.warm_start is not None:
      pb.warm_start = self.warm_start
    return pb
  def _from_proto(self, pb):
    if pb.WhichOneof("optional_apply_default_optimizations") is not None:
      self.apply_default_optimizations = pb.apply_default_optimizations
    if pb.WhichOneof("optional_filter_fusion") is not None:
      self.filter_fusion = pb.filter_fusion
    if pb.WhichOneof("optional_filter_parallelization") is not None:
      self.filter_parallelization = pb.filter_parallelization
    if pb.WhichOneof("optional_inject_prefetch") is not None:
      self.inject_prefetch = pb.inject_prefetch
    if pb.WhichOneof("optional_map_and_batch_fusion") is not None:
      self.map_and_batch_fusion = pb.map_and_batch_fusion
    if pb.WhichOneof("optional_map_and_filter_fusion") is not None:
      self.map_and_filter_fusion = pb.map_and_filter_fusion
    if pb.WhichOneof("optional_map_fusion") is not None:
      self.map_fusion = pb.map_fusion
    if pb.WhichOneof("optional_map_parallelization") is not None:
      self.map_parallelization = pb.map_parallelization
    if pb.WhichOneof("optional_noop_elimination") is not None:
      self.noop_elimination = pb.noop_elimination
    if pb.WhichOneof("optional_parallel_batch") is not None:
      self.parallel_batch = pb.parallel_batch
    if pb.WhichOneof("optional_shuffle_and_repeat_fusion") is not None:
      self.shuffle_and_repeat_fusion = pb.shuffle_and_repeat_fusion
    if pb.WhichOneof("optional_warm_start") is not None:
      self.warm_start = pb.warm_start
  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
