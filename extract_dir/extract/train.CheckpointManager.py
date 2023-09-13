@tf_export("train.CheckpointManager")
class CheckpointManager(object):
  """Manages multiple checkpoints by keeping some and deleting unneeded ones.
  Example usage:
  ```python
  import tensorflow as tf
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(
      checkpoint, directory="/tmp/model", max_to_keep=5)
  status = checkpoint.restore(manager.latest_checkpoint)
  while True:
    # train
    manager.save()
  ```
  `CheckpointManager` preserves its own state across instantiations (see the
  `__init__` documentation for details). Only one should be active in a
  particular directory at a time.
  """
  def __init__(self,
               checkpoint,
               directory,
               max_to_keep,
               keep_checkpoint_every_n_hours=None,
               checkpoint_name="ckpt",
               step_counter=None,
               checkpoint_interval=None,
               init_fn=None):
    """Configure a `CheckpointManager` for use in `directory`.
    If a `CheckpointManager` was previously used in `directory`, its
    state will be restored. This includes the list of managed checkpoints and
    the timestamp bookkeeping necessary to support
    `keep_checkpoint_every_n_hours`. The behavior of the new `CheckpointManager`
    will be the same as the previous `CheckpointManager`, including cleaning up
    existing checkpoints if appropriate.
    Checkpoints are only considered for deletion just after a new checkpoint has
    been added. At that point, `max_to_keep` checkpoints will remain in an
    "active set". Once a checkpoint is preserved by
    `keep_checkpoint_every_n_hours` it will not be deleted by this
    `CheckpointManager` or any future `CheckpointManager` instantiated in
    `directory` (regardless of the new setting of
    `keep_checkpoint_every_n_hours`). The `max_to_keep` checkpoints in the
    active set may be deleted by this `CheckpointManager` or a future
    `CheckpointManager` instantiated in `directory` (subject to its
    `max_to_keep` and `keep_checkpoint_every_n_hours` settings).
    `CheckpointManager` can be also used for initializing the model if
    there is no checkpoints for restoring in `directory`. An example usage is:
    >>> import tempfile
    >>> tmp_dir = tempfile.mkdtemp()
    >>> checkpoint = tf.train.Checkpoint()
    >>> init_path = checkpoint.save(os.path.join(tmp_dir, 'init'))
    >>> def init_fn():
    ...   # Partially restore the checkpoint from `init_path`.
    ...   checkpoint.restore(init_path)
    >>> manager = tf.train.CheckpointManager(
    ...     checkpoint,
    ...     directory=os.path.join(tmp_dir, 'ckpt'),
    ...     max_to_keep=None,
    ...     init_fn=init_fn)
    >>> # `restore_or_initialize` will call `init_fn` if there is no existing
    >>> # checkpoint in `directory`.
    >>> manager.restore_or_initialize()
    Args:
      checkpoint: The `tf.train.Checkpoint` instance to save and manage
        checkpoints for.
      directory: The path to a directory in which to write checkpoints. A
        special file named "checkpoint" is also written to this directory (in a
        human-readable text format) which contains the state of the
        `CheckpointManager`.
      max_to_keep: An integer, the number of checkpoints to keep. Unless
        preserved by `keep_checkpoint_every_n_hours`, checkpoints will be
        deleted from the active set, oldest first, until only `max_to_keep`
        checkpoints remain. If `None`, no checkpoints are deleted and everything
        stays in the active set. Note that `max_to_keep=None` will keep all
        checkpoint paths in memory and in the checkpoint state protocol buffer
        on disk.
      keep_checkpoint_every_n_hours: Upon removal from the active set, a
        checkpoint will be preserved if it has been at least
        `keep_checkpoint_every_n_hours` since the last preserved checkpoint. The
        default setting of `None` does not preserve any checkpoints in this way.
      checkpoint_name: Custom name for the checkpoint file.
      step_counter: A `tf.Variable` instance for checking the current step
        counter value, in case users want to save checkpoints every N steps.
      checkpoint_interval: An integer, indicates the minimum step interval
        between two checkpoints.
      init_fn: Callable. A function to do customized intialization if no
        checkpoints are in the directory.
    Raises:
      ValueError: If `max_to_keep` is not a positive integer.
    """
    self._checkpoint = checkpoint
    self._save_counter_assign = None
    if max_to_keep is not None and max_to_keep <= 0:
      raise ValueError(
          ("Expected a positive integer or `None` for `max_to_keep`, "
           "got %d.")
          % (max_to_keep,))
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    if isinstance(directory, os.PathLike):
      directory = os.fspath(directory)
    self._directory = directory
    self._checkpoint_prefix = os.path.join(directory, checkpoint_name)
    self._init_fn = init_fn
    if checkpoint_interval is not None:
      if step_counter is None:
        raise ValueError("`step_counter` should be passed if "
                         "`checkpoint_interval` is not None.")
      self._last_checkpoint_step = None
      self._step_counter = step_counter
    self._checkpoint_interval = checkpoint_interval
    recovered_state = get_checkpoint_state(directory)
    current_clock = time.time()
    self._maybe_delete = collections.OrderedDict()
    if recovered_state is None:
      self._latest_checkpoint = None
      # Set the clock back slightly to avoid race conditions when quickly
      # re-creating a CheckpointManager.
      self._last_preserved_timestamp = current_clock - 1.
    else:
      self._latest_checkpoint = recovered_state.model_checkpoint_path
      self._last_preserved_timestamp = recovered_state.last_preserved_timestamp
      if current_clock < self._last_preserved_timestamp:
        # Time seems to have reversed itself. In addition to this warning, we'll
        # min() saved checkpoint timestamps with the current time to ensure that
        # old checkpoints don't get deleted accidentally.
        logging.warning(
            ("time.time() returned a value %f seconds behind the last "
             "preserved checkpoint timestamp.")
            % (self._last_preserved_timestamp - current_clock,))
        self._last_preserved_timestamp = current_clock
      all_timestamps = recovered_state.all_model_checkpoint_timestamps
      all_paths = recovered_state.all_model_checkpoint_paths
      del recovered_state  # Uses modified values from now on
      if not all_timestamps:
        all_timestamps = [self._last_preserved_timestamp] * len(all_paths)
      for filename, timestamp in zip(all_paths, all_timestamps):
        timestamp = min(timestamp, current_clock)
        if timestamp > self._last_preserved_timestamp:
          self._maybe_delete[filename] = timestamp
  @property
  def directory(self):
    return self._directory
  @property
  def checkpoint_interval(self):
    return self._checkpoint_interval
  @property
  def latest_checkpoint(self):
    """The prefix of the most recent checkpoint in `directory`.
    Equivalent to `tf.train.latest_checkpoint(directory)` where `directory` is
    the constructor argument to `CheckpointManager`.
    Suitable for passing to `tf.train.Checkpoint.restore` to resume training.
    Returns:
      The checkpoint prefix. If there are no checkpoints, returns `None`.
    """
    return self._latest_checkpoint
  @property
  def checkpoints(self):
    """A list of managed checkpoints.
    Note that checkpoints saved due to `keep_checkpoint_every_n_hours` will not
    show up in this list (to avoid ever-growing filename lists).
    Returns:
      A list of filenames, sorted from oldest to newest.
    """
    return list(self._maybe_delete.keys())
  def _sweep(self):
    """Deletes or preserves managed checkpoints."""
    if not self._max_to_keep:
      # Does not update self._last_preserved_timestamp, since everything is kept
      # in the active set.
      return
    while len(self._maybe_delete) > self._max_to_keep:
      filename, timestamp = self._maybe_delete.popitem(last=False)
      # Even if we're keeping this checkpoint due to
      # keep_checkpoint_every_n_hours, we won't reference it to avoid
      # infinitely-growing CheckpointState protos.
      if (self._keep_checkpoint_every_n_hours
          and (timestamp - self._keep_checkpoint_every_n_hours * 3600.
               >= self._last_preserved_timestamp)):
        self._last_preserved_timestamp = timestamp
        continue
      _delete_file_if_exists(filename + ".index")
      _delete_file_if_exists(filename + ".data-?????-of-?????")
  def _record_state(self):
    """Saves the `CheckpointManager`'s state in `directory`."""
    filenames, timestamps = zip(*self._maybe_delete.items())
    update_checkpoint_state_internal(
        self._directory,
        model_checkpoint_path=self.latest_checkpoint,
        all_model_checkpoint_paths=filenames,
        all_model_checkpoint_timestamps=timestamps,
        last_preserved_timestamp=self._last_preserved_timestamp,
        save_relative_paths=True)
  @property
  def _prefix(self):
    """A common prefix for all checkpoints saved with this manager.
    For example, if `directory` (a constructor argument) were `"/tmp/tf-model"`,
    `prefix` would be `"/tmp/tf-model/ckpt"` and checkpoints would generally be
    numbered `"/tmp/tf-model/ckpt-1"`, `"/tmp/tf-model/ckpt-2"`, and so on. Each
    checkpoint has several associated files
    (e.g. `"/tmp/tf-model/ckpt-2.index"`).
    Returns:
      A string prefix.
    """
    return self._checkpoint_prefix
  @property
  def checkpoint(self):
    """Returns the `tf.train.Checkpoint` object."""
    return self._checkpoint
  def save(self, checkpoint_number=None, check_interval=True, options=None):
    """Creates a new checkpoint and manages it.
    Args:
      checkpoint_number: An optional integer, or an integer-dtype `Variable` or
        `Tensor`, used to number the checkpoint. If `None` (default),
        checkpoints are numbered using `checkpoint.save_counter`. Even if
        `checkpoint_number` is provided, `save_counter` is still incremented. A
        user-provided `checkpoint_number` is not incremented even if it is a
        `Variable`.
      check_interval: An optional boolean. The argument is only effective when
        `checkpoint_interval` is passed into the manager. If `True`, the manager
        will only save the checkpoint if the interval between checkpoints is
        larger than `checkpoint_interval`. Otherwise it will always save the
        checkpoint unless a checkpoint has already been saved for the current
        step.
      options: Optional `tf.train.CheckpointOptions` object. This argument only
        works with TF2 checkpoint objects. For example, options =
        tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    Returns:
      The path to the new checkpoint. It is also recorded in the `checkpoints`
      and `latest_checkpoint` properties. `None` if no checkpoint is saved.
    """
    if self._checkpoint_interval is not None:
      current_step = _evaluate(self._step_counter)
      if self._last_checkpoint_step is not None:
        if current_step == self._last_checkpoint_step:
          return None
        if check_interval and current_step < (
            self._last_checkpoint_step + self._checkpoint_interval):
          return None
      self._last_checkpoint_step = current_step
    # Save counter logic duplicated from tf.train.Checkpoint, soon to diverge
    # slightly with a custom numbering option.
    if context.executing_eagerly():
      save_counter = self._checkpoint.save_counter
      save_counter.assign_add(1)
      session = None
    else:
      session = ops.get_default_session()
      def _initializing_creator(next_creator, **kwargs):
        """Initialize the save counter if it has been newly created."""
        v = next_creator(**kwargs)
        session.run(v.initializer)
        return v
      with variable_scope.variable_creator_scope(_initializing_creator):
        save_counter = self._checkpoint.save_counter
      if self._save_counter_assign is None:
        self._save_counter_assign = save_counter.assign_add(1, read_value=False)
      session.run(self._save_counter_assign)
    if checkpoint_number is None:
      checkpoint_number = save_counter
    if not isinstance(checkpoint_number, compat.integral_types):
      checkpoint_number = training_util.global_step(
          sess=session, global_step_tensor=checkpoint_number)
    prefix = "%s-%d" % (self._prefix, checkpoint_number)
    def _record_and_sweep_state(save_path):
      timestamp = time.time()
      # If this is an overwritten checkpoint we were previously tracking, delete
      # and reinsert it to make sure it goes to the end of the queue.
      if save_path in self._maybe_delete:
        del self._maybe_delete[save_path]
      self._maybe_delete[save_path] = timestamp
      self._latest_checkpoint = save_path
      # Before deleting anything we update the Checkpoint proto with the new
      # checkpoint. We'll go back and correct it after cleaning up old files,
      # but a preemption while deleting will be more likely to see the new
      # checkpoint this way.
      self._record_state()
      self._sweep()
      # Write out the Checkpoint proto a second time, now without the deleted
      # checkpoints.
      self._record_state()
    if options is None:
      save_path = self._checkpoint._write(  # pylint: disable=protected-access
          prefix, write_done_callback=_record_and_sweep_state)
    else:
      save_path = self._checkpoint._write(  # pylint: disable=protected-access
          prefix, options=options, write_done_callback=_record_and_sweep_state)
    return save_path
  def restore_or_initialize(self):
    """Restore items in `checkpoint` from the latest checkpoint file.
    This method will first try to restore from the most recent checkpoint in
    `directory`. If no checkpoints exist in `directory`, and `init_fn` is
    specified, this method will call `init_fn` to do customized
    initialization. This can be used to support initialization from pretrained
    models.
    Note that unlike `tf.train.Checkpoint.restore()`, this method doesn't return
    a load status object that users can run assertions on
    (e.g. assert_consumed()). Thus to run assertions, users should directly use
    `tf.train.Checkpoint.restore()` method.
    Returns:
      The restored checkpoint path if the lastest checkpoint is found and
      restored. Otherwise None.
    """
    # TODO(chienchunh): When AsyncCheckpoint is used, we may need to force to
    # sync until any ongoing async save is done. Otherwise, if this is the first
    # checkpoint and _latest_checkpoint has not been updated due to async write,
    # this would resort to init_fn instead of restoring from the checkpoin file.
    # This should be fixed once AsyncCheckpoint is integrated with the public
    # API so that we can rely on CheckpointOptions to tell whether we should
    # sync for AsyncCheckpoint.
    if self._latest_checkpoint is not None:
      self._checkpoint.restore(self._latest_checkpoint)
      if self._checkpoint_interval is not None:
        self._last_checkpoint_step = _evaluate(self._step_counter)
      return self._latest_checkpoint
    if self._init_fn is not None:
      self._init_fn()
      logging.info(
          "Customized initialization is done through the passed `init_fn`.")
    return None
  def sync(self):
    """Wait for any outstanding save or restore operations."""
    if self._checkpoint:
      self._checkpoint.sync()
