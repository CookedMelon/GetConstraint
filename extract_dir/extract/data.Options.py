@tf_export("data.Options")
class Options(options_lib.OptionsBase):
  """Represents options for `tf.data.Dataset`.
  A `tf.data.Options` object can be, for instance, used to control which static
  optimizations to apply to the input pipeline graph or whether to use
  performance modeling to dynamically tune the parallelism of operations such as
  `tf.data.Dataset.map` or `tf.data.Dataset.interleave`.
  The options are set for the entire dataset and are carried over to datasets
  created through tf.data transformations.
  The options can be set by constructing an `Options` object and using the
  `tf.data.Dataset.with_options(options)` transformation, which returns a
  dataset with the options set.
  >>> dataset = tf.data.Dataset.range(42)
  >>> options = tf.data.Options()
  >>> options.deterministic = False
  >>> dataset = dataset.with_options(options)
  >>> print(dataset.options().deterministic)
  False
  Note: A known limitation of the `tf.data.Options` implementation is that the
  options are not preserved across tf.function boundaries. In particular, to
  set options for a dataset that is iterated within a tf.function, the options
  need to be set within the same tf.function.
  """
  autotune = options_lib.create_option(
      name="autotune",
      ty=AutotuneOptions,
      docstring="The autotuning options associated with the dataset. See "
      "`tf.data.experimental.AutotuneOptions` for more details.",
      default_factory=AutotuneOptions)
  deterministic = options_lib.create_option(
      name="deterministic",
      ty=bool,
      docstring=
      "Whether the outputs need to be produced in deterministic order. If None,"
      " defaults to True.")
  experimental_deterministic = options_lib.create_option(
      name="experimental_deterministic",
      ty=bool,
      docstring="DEPRECATED. Use `deterministic` instead.")
  experimental_distribute = options_lib.create_option(
      name="experimental_distribute",
      ty=DistributeOptions,
      docstring=
      "The distribution strategy options associated with the dataset. See "
      "`tf.data.experimental.DistributeOptions` for more details.",
      default_factory=DistributeOptions)
  experimental_external_state_policy = options_lib.create_option(
      name="experimental_external_state_policy",
      ty=ExternalStatePolicy,
      docstring="This option can be used to override the default policy for "
      "how to handle external state when serializing a dataset or "
      "checkpointing its iterator. There are three settings available - "
      "IGNORE: External state is ignored without a warning; WARN: External "
      "state is ignored and a warning is logged; FAIL: External state results "
      "in an error.")
  experimental_optimization = options_lib.create_option(
      name="experimental_optimization",
      ty=OptimizationOptions,
      docstring=
      "The optimization options associated with the dataset. See "
      "`tf.data.experimental.OptimizationOptions` for more details.",
      default_factory=OptimizationOptions)
  experimental_slack = options_lib.create_option(
      name="experimental_slack",
      ty=bool,
      docstring="Whether to introduce 'slack' in the last `prefetch` of the "
      "input pipeline, if it exists. This may reduce CPU contention with "
      "accelerator host-side activity at the start of a step. The slack "
      "frequency is determined by the number of devices attached to this "
      "input pipeline. If None, defaults to False.")
  experimental_symbolic_checkpoint = options_lib.create_option(
      name="experimental_symbolic_checkpoint",
      ty=bool,
      docstring="Whether to checkpoint internal input pipeline state "
      "maintaining cursors into data sources that identify last "
      "element(s) produced as output to the tf.data consumer. This "
      "is alternative to the default 'explicit' checkpointing which "
      "stores the internal input pipeline state in the checkpoint. "
      "Note that symbolic checkpointing is not supported for "
      "transformations that can reorder elements.")
  experimental_threading = options_lib.create_option(
      name="experimental_threading",
      ty=ThreadingOptions,
      docstring="DEPRECATED. Use `threading` instead.")
  threading = options_lib.create_option(
      name="threading",
      ty=ThreadingOptions,
      docstring="The threading options associated with the dataset. See "
      "`tf.data.ThreadingOptions` for more details.",
      default_factory=ThreadingOptions)
  def __getattribute__(self, name):
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      return getattr(self, "threading")
    if name == "experimental_deterministic":
      # TODO(aaudibert): Uncomment after internal uses have been updated.
      # logging.warning("options.experimental_deterministic is deprecated. "
      #                 "Use options.deterministic instead.")
      return getattr(self, "deterministic")
    return super(Options, self).__getattribute__(name)
  def __setattr__(self, name, value):
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      super(Options, self).__setattr__("threading", value)
      return
    if name == "experimental_deterministic":
      # TODO(aaudibert): Uncomment after internal uses have been updated.
      # logging.warning("options.experimental_deterministic is deprecated. "
      #                 "Use options.deterministic instead.")
      super(Options, self).__setattr__("deterministic", value)
      return
    if name == "experimental_symbolic_checkpoint":
      # TODO(b/276269493): Add support for MacOS.
      if platform.system() == "Darwin":
        logging.warning("Symbolic checkpointing is not supported on MacOS.")
        return
    super(Options, self).__setattr__(name, value)
  def _to_proto(self):
    pb = dataset_options_pb2.Options()
    if self.deterministic is not None:
      pb.deterministic = self.deterministic
    pb.autotune_options.CopyFrom(self.autotune._to_proto())  # pylint: disable=protected-access
    pb.distribute_options.CopyFrom(self.experimental_distribute._to_proto())  # pylint: disable=protected-access
    if self.experimental_external_state_policy is not None:
      pb.external_state_policy = (
          ExternalStatePolicy._to_proto(  # pylint: disable=protected-access
              self.experimental_external_state_policy))
    pb.optimization_options.CopyFrom(self.experimental_optimization._to_proto())  # pylint: disable=protected-access
    if self.experimental_slack is not None:
      pb.slack = self.experimental_slack
    if self.experimental_symbolic_checkpoint is not None:
      pb.symbolic_checkpoint = self.experimental_symbolic_checkpoint
    pb.threading_options.CopyFrom(self.threading._to_proto())  # pylint: disable=protected-access
    return pb
  def _from_proto(self, pb):
    if pb.WhichOneof("optional_deterministic") is not None:
      self.deterministic = pb.deterministic
    self.autotune._from_proto(pb.autotune_options)  # pylint: disable=protected-access
    self.experimental_distribute._from_proto(pb.distribute_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_external_state_policy") is not None:
      self.experimental_external_state_policy = (
          ExternalStatePolicy._from_proto(  # pylint: disable=protected-access
              pb.external_state_policy))
    self.experimental_optimization._from_proto(pb.optimization_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_slack") is not None:
      self.experimental_slack = pb.slack
    if pb.WhichOneof("optional_symbolic_checkpoint") is not None:
      self.experimental_symbolic_checkpoint = pb.symbolic_checkpoint
    self.threading._from_proto(pb.threading_options)  # pylint: disable=protected-access
  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
    self.autotune._set_mutable(mutable)
    self.experimental_distribute._set_mutable(mutable)
    self.experimental_optimization._set_mutable(mutable)
    self.threading._set_mutable(mutable)
  def merge(self, options):
    """Merges itself with the given `tf.data.Options`.
    If this object and the `options` to merge set an option differently, a
    warning is generated and this object's value is updated with the `options`
    object's value.
    Args:
      options: The `tf.data.Options` to merge with.
    Returns:
      New `tf.data.Options` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)
