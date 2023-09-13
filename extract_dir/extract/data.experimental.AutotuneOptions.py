@tf_export("data.experimental.AutotuneOptions")
class AutotuneOptions(options_lib.OptionsBase):
  """Represents options for autotuning dataset performance.
  ```python
  options = tf.data.Options()
  options.autotune.enabled = False
  dataset = dataset.with_options(options)
  ```
  """
  enabled = options_lib.create_option(
      name="enabled",
      ty=bool,
      docstring="Whether to automatically tune performance knobs. If None, "
      "defaults to True.")
  cpu_budget = options_lib.create_option(
      name="cpu_budget",
      ty=int,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the CPU budget to use. Values greater than the number of schedulable "
      "CPU cores are allowed but may result in CPU contention. If None, "
      "defaults to the number of schedulable CPU cores.")
  ram_budget = options_lib.create_option(
      name="ram_budget",
      ty=int,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the RAM budget to use. Values greater than the available RAM in bytes "
      "may result in OOM. If None, defaults to half of the available RAM in "
      "bytes.")
  autotune_algorithm = options_lib.create_option(
      name="autotune_algorithm",
      ty=AutotuneAlgorithm,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the algorithm to use.")
  def _to_proto(self):
    pb = dataset_options_pb2.AutotuneOptions()
    if self.enabled is not None:
      pb.enabled = self.enabled
    if self.cpu_budget is not None:
      pb.cpu_budget = self.cpu_budget
    if self.ram_budget is not None:
      pb.ram_budget = self.ram_budget
    if self.autotune_algorithm is not None:
      pb.autotune_algorithm = AutotuneAlgorithm._to_proto(  # pylint: disable=protected-access
          self.autotune_algorithm)
    return pb
  def _from_proto(self, pb):
    if pb.WhichOneof("optional_enabled") is not None:
      self.enabled = pb.enabled
    if pb.WhichOneof("optional_cpu_budget") is not None:
      self.cpu_budget = pb.cpu_budget
    if pb.WhichOneof("optional_ram_budget") is not None:
      self.ram_budget = pb.ram_budget
    if pb.WhichOneof("optional_autotune_algorithm") is not None:
      self.autotune_algorithm = AutotuneAlgorithm._from_proto(  # pylint: disable=protected-access
          pb.autotune_algorithm)
  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
