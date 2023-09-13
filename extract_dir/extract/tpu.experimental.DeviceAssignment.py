@tf_export("tpu.experimental.DeviceAssignment")
class DeviceAssignment(object):
  """Mapping from logical cores in a computation to the physical TPU topology.
  Prefer to use the `DeviceAssignment.build()` helper to construct a
  `DeviceAssignment`; it is easier if less flexible than constructing a
  `DeviceAssignment` directly.
  """
  def __init__(self, topology: Topology, core_assignment: np.ndarray):
    """Constructs a `DeviceAssignment` object.
    Args:
      topology: A `Topology` object that describes the physical TPU topology.
      core_assignment: A logical to physical core mapping, represented as a
        rank 3 numpy array. See the description of the `core_assignment`
        property for more details.
    Raises:
      ValueError: If `topology` is not `Topology` object.
      ValueError: If `core_assignment` is not a rank 3 numpy array.
    """
    if not isinstance(topology, Topology):
      raise ValueError("topology must be a Topology object, got {}".format(
          type(topology)))
    core_assignment = np.asarray(core_assignment, dtype=np.int32)
    self._topology = topology
    if core_assignment.ndim != 3:
      raise ValueError("core_assignment must be a rank 3 numpy array, "
                       f"got shape {core_assignment.shape}")
    self._num_replicas = core_assignment.shape[0]
    self._num_cores_per_replica = core_assignment.shape[1]
    if core_assignment.shape[-1] != topology.mesh_rank:
      raise ValueError(
          "core_assignment.shape[-1] must have size equal to topology "
          f"rank ({topology.mesh_rank}), got "
          f"core_assignment.shape={core_assignment.shape}")
    self._core_assignment = core_assignment
    self._task_and_cores_to_replicas = _compute_task_and_cores_to_replicas(
        self._core_assignment, topology)
  @property
  def topology(self) -> Topology:
    """A `Topology` that describes the TPU topology."""
    return self._topology
  @property
  def num_cores_per_replica(self) -> int:
    """The number of cores per replica."""
    return self._num_cores_per_replica
  @property
  def num_replicas(self) -> int:
    """The number of replicas of the computation."""
    return self._num_replicas
  @property
  def core_assignment(self) -> np.ndarray:
    """The logical to physical core mapping.
    Returns:
      An integer numpy array of rank 3, with shape
      `[num_replicas, num_cores_per_replica, topology_rank]`. Maps
      (replica, logical core) pairs to physical topology coordinates.
    """
    return self._core_assignment
  def coordinates(self, replica: int, logical_core: int) -> Tuple:  # pylint:disable=g-bare-generic
    """Returns the physical topology coordinates of a logical core."""
    return tuple(self.core_assignment[replica, logical_core, :])
  def lookup_replicas(self, task_id: int, logical_core: int) -> List[int]:
    """Lookup replica ids by task number and logical core.
    Args:
      task_id: TensorFlow task number.
      logical_core: An integer, identifying a logical core.
    Returns:
      A sorted list of the replicas that are attached to that task and
      logical_core.
    Raises:
      ValueError: If no replica exists in the task which contains the logical
      core.
    """
    try:
      return self._task_and_cores_to_replicas[task_id][logical_core]
    except KeyError:
      raise ValueError(
          "Can not find any replica in task: {} contains logical_core: {} ".
          format(task_id, logical_core))
  def tpu_ordinal(self, replica: int = 0, logical_core: int = 0) -> int:
    """Returns the ordinal of the TPU device assigned to a logical core."""
    coordinates = self.coordinates(replica, logical_core)
    return self._topology.tpu_device_ordinal_at_coordinates(coordinates)
  def host_device(self,
                  replica: int = 0,
                  logical_core: int = 0,
                  job: Optional[Text] = None) -> Text:
    """Returns the CPU device attached to a logical core."""
    coordinates = self.coordinates(replica, logical_core)
    return self._topology.cpu_device_name_at_coordinates(coordinates, job=job)
  def tpu_device(self,
                 replica: int = 0,
                 logical_core: int = 0,
                 job: Optional[Text] = None) -> Text:
    """Returns the name of the TPU device assigned to a logical core."""
    coordinates = self.coordinates(replica, logical_core)
    return self._topology.tpu_device_name_at_coordinates(coordinates, job=job)
  @staticmethod
  def build(topology: Topology,
            computation_shape: Optional[np.ndarray] = None,
            computation_stride: Optional[np.ndarray] = None,
            num_replicas: int = 1) -> "DeviceAssignment":
    return device_assignment(topology, computation_shape, computation_stride,
                             num_replicas)
