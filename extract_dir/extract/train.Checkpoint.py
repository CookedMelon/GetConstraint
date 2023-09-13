@tf_export("train.Checkpoint", v1=[])
class Checkpoint(autotrackable.AutoTrackable):
  """Manages saving/restoring trackable values to disk.
  TensorFlow objects may contain trackable state, such as `tf.Variable`s,
  `tf.keras.optimizers.Optimizer` implementations, `tf.data.Dataset` iterators,
  `tf.keras.Layer` implementations, or  `tf.keras.Model` implementations.
  These are called **trackable objects**.
  A `Checkpoint` object can be constructed to save either a single or group of
  trackable objects to a checkpoint file. It maintains a `save_counter` for
  numbering checkpoints.
  Example:
  ```python
  model = tf.keras.Model(...)
  checkpoint = tf.train.Checkpoint(model)
  # Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
  # checkpoint.save is called, the save counter is increased.
  save_path = checkpoint.save('/tmp/training_checkpoints')
  # Restore the checkpointed values to the `model` object.
  checkpoint.restore(save_path)
  ```
  Example 2:
  ```python
  import tensorflow as tf
  import os
  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  # Create a Checkpoint that will manage two objects with trackable state,
  # one we name "optimizer" and the other we name "model".
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )  # Variables will be restored on creation.
  status.assert_consumed()  # Optional sanity checks.
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```
  `Checkpoint.save()` and `Checkpoint.restore()` write and read object-based
  checkpoints, in contrast to TensorFlow 1.x's `tf.compat.v1.train.Saver` which
  writes and
  reads `variable.name` based checkpoints. Object-based checkpointing saves a
  graph of dependencies between Python objects (`Layer`s, `Optimizer`s,
  `Variable`s, etc.) with named edges, and this graph is used to match variables
  when restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables.
  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their own variables (e.g. "kernel" and "bias" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:
  ```python
  class Regress(tf.keras.Model):
    def __init__(self):
      super().__init__()
      self.input_transform = tf.keras.layers.Dense(10)
      # ...
    def call(self, inputs):
      x = self.input_transform(inputs)
      # ...
  ```
  This `Model` has a dependency named "input_transform" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.
  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.
  This function differs slightly from the Keras Model `save_weights` function.
  `tf.keras.Model.save_weights` creates a checkpoint file with the name
  specified in `filepath`, while `tf.train.Checkpoint` numbers the checkpoints,
  using `filepath` as the prefix for the checkpoint file names. Aside from this,
  `model.save_weights()` and `tf.train.Checkpoint(model).save()` are equivalent.
  See the [guide to training
  checkpoints](https://www.tensorflow.org/guide/checkpoint) for
  details.
  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """
  def __init__(self, root=None, **kwargs):
    """Creates a training checkpoint for a single or group of objects.
    Args:
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. All `kwargs` must be trackable objects, or a
        nested structure of trackable objects (`list`, `dict`, or `tuple`).
    Raises:
      ValueError: If `root` or the objects in `kwargs` are not trackable. A
        `ValueError` is also raised if the `root` object tracks different
        objects from the ones listed in attributes in kwargs (e.g.
        `root.child = A` and `tf.train.Checkpoint(root, child=B)` are
        incompatible).
    """
    super().__init__()
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()
    # Store a reference to root and kwargs if we need to instantiate an
    # AsyncCheckpointer later.
    self._root = root
    self._kwargs = kwargs
    self._delete_tracking("_kwargs")
    # Don't instantiate the AsyncCheckpointer unless required.
    self._async_checkpointer_impl = None
    # Store checkpoint options during the save/write calls so that subsequent
    # read/restore calls are done properly. This is only populated when
    # async read/write is enabled.
    self._checkpoint_options = None
    attached_dependencies = None
    self._save_counter = None  # Created lazily for restore-on-create.
    self._save_assign_op = None
    if root:
      trackable_root = root() if isinstance(root, weakref.ref) else root
      _assert_trackable(trackable_root, "root")
      attached_dependencies = []
      # All keyword arguments (including root itself) are set as children
      # of root.
      kwargs["root"] = root
      trackable_root._maybe_initialize_trackable()
      self._save_counter = data_structures.NoDependency(
          trackable_root._lookup_dependency("save_counter"))
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)
      # Call getattr instead of directly using v because setattr converts
      # v to a Trackable data structure when v is a list/dict/tuple.
      converted_v = getattr(self, k)
      if isinstance(converted_v, weakref.ref):
        converted_v = converted_v()
      _assert_trackable(converted_v, k)
      if root:
        # Make sure that root doesn't already have dependencies with these names
        child = trackable_root._lookup_dependency(k)
        if child is None:
          attached_dependencies.append(
              base.WeakTrackableReference(k, converted_v))
        elif child != converted_v:
          raise ValueError(
              f"Cannot create a Checkpoint with keyword argument {k} if "
              f"root.{k} already exists.")
    self._saver = TrackableSaver(
        graph_view_lib.ObjectGraphView(
            root if root else self,
            attached_dependencies=attached_dependencies))
    self._attached_dependencies = data_structures.NoDependency(
        attached_dependencies)
  def _maybe_create_save_counter(self):
    """Create a save counter if it does not yet exist."""
    if self._save_counter is None:
      # Initialized to 0 and incremented before saving.
      with ops.device("/cpu:0"):
        # add_variable creates a dependency named "save_counter"; NoDependency
        # prevents creating a second dependency named "_save_counter".
        self._save_counter = data_structures.NoDependency(
            add_variable(
                self,
                name="save_counter",
                initializer=0,
                dtype=dtypes.int64,
                trainable=False))
        if self._attached_dependencies is not None:
          self._attached_dependencies.append(
              # Store a stronge reference to the `save_counter`, so that if the
              # `Checkpoint` object is deleted, the `save_counter` does not get
              # deleted immediately. (The LoadStatus object needs to indirectly
              # reference the counter through the ObjectGraphView).
              base.TrackableReference("save_counter", self._save_counter))
          # When loading a checkpoint, the save counter is created after
          # the checkpoint has been loaded, so it must be handled in a deferred
          # manner.
          if isinstance(self.root, weakref.ref):
            root = self.root()
          else:
            root = self.root
          restore = root._deferred_dependencies.pop("save_counter", ())  # pylint: disable=protected-access
          if restore:
            restore[0].restore(self._save_counter)
  def write(self, file_prefix, options=None):
    """Writes a training checkpoint.
    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.
    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.
    Checkpoints written with `write` must be read with `read`.
    Example usage:
    ```
    step = tf.Variable(0, name="step")
    checkpoint = tf.Checkpoint(step=step)
    checkpoint.write("/tmp/ckpt")
    # Later, read the checkpoint with read()
    checkpoint.read("/tmp/ckpt")
    # You can also pass options to write() and read(). For example this
    # runs the IO ops on the localhost:
    options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.write("/tmp/ckpt", options=options)
    # Later, read the checkpoint with read()
    checkpoint.read("/tmp/ckpt", options=options)
    ```
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    if isinstance(file_prefix, os.PathLike):
      file_prefix = os.fspath(file_prefix)
    return self._write(file_prefix, options)
  def _async_checkpointer(self):
    """Returns an instantiated AsyncCheckpointHelper."""
    if self._async_checkpointer_impl is None:
      self._async_checkpointer_impl = (
          async_checkpoint_helper.AsyncCheckpointHelper(
              Checkpoint,
              **self._kwargs))
    return self._async_checkpointer_impl
  def _write(self, file_prefix, options=None, write_done_callback=None):
    """Internal method that implements Checkpoint.write().
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.
      write_done_callback: Optional callback function to be executed once
        the underlying checkpoint saving is finished. Example usage includes
        updating the checkpoint internal state.
    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    # Triggers TF2 async checkpoint handling if:
    # 1. async checkpoint is enabled in CheckpointOptions
    # 2. running in eager mode
    if options and options.experimental_enable_async_checkpoint:
      self._checkpoint_options = options
      if checkpoint_context.in_preemption_save_context():
        # Make sure all in-progress writes have completed before saving the
        # final preemption checkpoint.
        if self._async_checkpointer_impl is not None:
          self._async_checkpointer_impl.sync()
        # Additional work done will not be saved in a future checkpoint, so
        # we use regular sync checkpoint to avoid overhead of dispatching
        # checkpoint write to a new thread.
        logging.warning(
            "Switching to regular sync checkpoint for preemption checkpoint."
        )
      elif context.executing_eagerly():
        return self._async_checkpointer()._write(  # pylint: disable=protected-access
            file_prefix, options, write_done_callback)
      else:
        logging.warning(
            "Saving async checkpoint in graph mode is currently not supported;"
            " switching to regular sync checkpoint instead.")
    start_time = time.time()
    options = options or checkpoint_options.CheckpointOptions()
    output = self._saver.save(file_prefix=file_prefix, options=options)
    output = _convert_file_name_tensor_to_string(output)
    if write_done_callback:
      write_done_callback(output)
    # Ensure save operations have completed when running in eager runtime.
    if context.executing_eagerly():
      context.async_wait()
    end_time = time.time()
    if not checkpoint_context.in_async_metrics_context():
      # This records the time checkpoint._write() blocks on the main thread.
      metrics.AddCheckpointWriteDuration(
          api_label=_CHECKPOINT_V2,
          microseconds=_get_duration_microseconds(start_time, end_time),
      )
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if not checkpoint_context.in_async_metrics_context():
        metrics.AddTrainingTimeSaved(
            api_label=_CHECKPOINT_V2,
            microseconds=_get_duration_microseconds(
                _END_TIME_OF_LAST_WRITE, end_time)
        )
      if checkpoint_context.in_preemption_save_context():
        _preemption_checkpoint_saved_time_usecs.get_cell().increase_by(
            _get_duration_microseconds(_END_TIME_OF_LAST_WRITE, end_time)
        )
      _END_TIME_OF_LAST_WRITE = end_time
    metrics.RecordCheckpointSize(
        api_label=_CHECKPOINT_V2, filesize=_get_checkpoint_size(output)
    )
    return output
  @property
  def save_counter(self):
    """An integer variable which starts at zero and is incremented on save.
    Used to number checkpoints.
    Returns:
      The save counter variable.
    """
    self._maybe_create_save_counter()
    return self._save_counter
  def sync(self):
    """Wait for any outstanding save or restore operations."""
    # Subclasses of Checkpoint may not have `_async_checkpointer_impl` so use
    # `getattr` for safer check.
    if getattr(self, "_async_checkpointer_impl", None) is not None:
      self._async_checkpointer_impl.sync()
  def save(self, file_prefix, options=None):
    # pylint:disable=line-too-long
    """Saves a training checkpoint and provides basic checkpoint management.
    The saved checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.save()` is
    called.
    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write` and `read`.
    (`tf.train.CheckpointManager` for example).
    ```
    step = tf.Variable(0, name="step")
    checkpoint = tf.train.Checkpoint(step=step)
    checkpoint.save("/tmp/ckpt")
    # Later, read the checkpoint with restore()
    checkpoint.restore("/tmp/ckpt-1")
    # You can also pass options to save() and restore(). For example this
    # runs the IO ops on the localhost:
    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.save("/tmp/ckpt", options=options)
    # Later, read the checkpoint with restore()
    checkpoint.restore("/tmp/ckpt-1", options=options)
    ```
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      The full path to the checkpoint.
    """
    # Triggers TF2 async checkpoint handling if:
    # 1. async checkpoint is enabled in CheckpointOptions
    # 2. running in eager mode
    if options and options.experimental_enable_async_checkpoint:
      self._checkpoint_options = options
      if checkpoint_context.in_preemption_save_context():
        # Make sure all in-progress writes have completed before saving the
        # final preemption checkpoint.
        if self._async_checkpointer_impl is not None:
          self._async_checkpointer_impl.sync()
        # Additional work done will not be saved in a future checkpoint, so
        # we use regular sync checkpoint to avoid overhead of dispatching
        # checkpoint write to a new thread.
        logging.warning(
            "Switching to regular sync checkpoint for preemption checkpoint."
        )
      elif context.executing_eagerly():
        return self._async_checkpointer().save(file_prefix, options)
      else:
        logging.warning(
            "Saving async checkpoint in graph mode is currently not supported;"
            " switching to regular sync checkpoint instead.")
    if isinstance(file_prefix, os.PathLike):
      file_prefix = os.fspath(file_prefix)
    # pylint:enable=line-too-long
    options = options or checkpoint_options.CheckpointOptions()
    graph_building = not context.executing_eagerly()
    if graph_building:
      if ops.inside_function():
        raise NotImplementedError(
            "Calling tf.train.Checkpoint.save() from a function is not "
            "supported, as save() modifies saving metadata in ways not "
            "supported by TensorFlow Operations. Consider using "
            "tf.train.Checkpoint.write(), a lower-level API which does not "
            "update metadata. tf.train.latest_checkpoint and related APIs will "
            "not see this checkpoint.")
      session = get_session()
      if self._save_counter is None:
        # When graph building, if this is a new save counter variable then it
        # needs to be initialized before assign_add. This is only an issue if
        # restore() has not been called first.
        session.run(self.save_counter.initializer)
    if not graph_building or self._save_assign_op is None:
      with ops.colocate_with(self.save_counter):
        assign_op = self.save_counter.assign_add(1, read_value=True)
      if graph_building:
        self._save_assign_op = data_structures.NoDependency(assign_op)
    if graph_building:
      checkpoint_number = session.run(self._save_assign_op)
    else:
      checkpoint_number = assign_op.numpy()
    return self._write(
        "%s-%d" % (file_prefix, checkpoint_number),
        options=options,
        write_done_callback=_update_checkpoint_state_internal)
  def read(self, save_path, options=None):
    """Reads a training checkpoint written with `write`.
    Reads this `Checkpoint` and any objects it depends on.
    This method is just like `restore()` but does not expect the `save_counter`
    variable in the checkpoint. It only restores the objects that the checkpoint
    already depends on.
    The method is primarily intended for use by higher level checkpoint
    management utilities that use `write()` instead of `save()` and have their
    own mechanisms to number and track checkpoints.
    Example usage:
    ```python
    # Create a checkpoint with write()
    ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
    path = ckpt.write('/tmp/my_checkpoint')
    # Later, load the checkpoint with read()
    # With restore() assert_consumed() would have failed.
    checkpoint.read(path).assert_consumed()
    # You can also pass options to read(). For example this
    # runs the IO ops on the localhost:
    options = tf.train.CheckpointOptions(
        experimental_io_device="/job:localhost")
    checkpoint.read(path, options=options)
    ```
    Args:
      save_path: The path to the checkpoint as returned by `write`.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.  See `restore` for details.
    """
    if options and options.experimental_enable_async_checkpoint:
      self._checkpoint_options = options
    # Triggers TF2 async checkpoint handling if:
    # 1. async checkpoint is enabled in CheckpointOptions
    # 2. there's a preceeding async save/write
    # 3. running in eager mode
    if (self._checkpoint_options and
        self._checkpoint_options.experimental_enable_async_checkpoint):
      if context.executing_eagerly():
        return self._async_checkpointer().read(save_path, options)
      else:
        logging.warning(
            "Saving async checkpoint in graph mode is currently not supported;"
            " switching to regular sync checkpoint instead.")
    start_time = time.time()
    if isinstance(save_path, os.PathLike):
      save_path = os.fspath(save_path)
    options = options or checkpoint_options.CheckpointOptions()
    result = self._saver.restore(save_path=save_path, options=options)
    metrics.AddCheckpointReadDuration(
        api_label=_CHECKPOINT_V2,
        microseconds=_get_duration_microseconds(start_time, time.time()))
    return result
  def restore(self, save_path, options=None):
    """Restores a training checkpoint.
    Restores this `Checkpoint` and any objects it depends on.
    This method is intended to be used to load checkpoints created by `save()`.
    For checkpoints created by `write()` use the `read()` method which does not
    expect the `save_counter` variable added by `save()`.
    `restore()` either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added after this call will be matched if they have a
    corresponding object in the checkpoint (the restore request will queue in
    any trackable object waiting for the expected dependency to be added).
    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path)
    # You can additionally pass options to restore():
    options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.restore(path, options=options)
    ```
    To ensure that loading is complete and no more deferred restorations will
    take place, use the `assert_consumed()` method of the status object returned
    by `restore()`:
    ```python
    checkpoint.restore(path, options=options).assert_consumed()
    ```
    The assert will raise an error if any Python objects in the dependency graph
    were not found in the checkpoint, or if any checkpointed values do not have
    a matching Python object.
    Name-based `tf.compat.v1.train.Saver` checkpoints from TensorFlow 1.x can be
    loaded using this method. Names are used to match variables. Re-encode
    name-based checkpoints using `tf.train.Checkpoint.save` as soon as possible.
    **Loading from SavedModel checkpoints**
    To load values from a SavedModel, just pass the SavedModel directory
    to checkpoint.restore:
    ```python
    model = tf.keras.Model(...)
    tf.saved_model.save(model, path)  # or model.save(path, save_format='tf')
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(path).expect_partial()
    ```
    This example calls `expect_partial()` on the loaded status, since
    SavedModels saved from Keras often generates extra keys in the checkpoint.
    Otherwise, the program prints a lot of warnings about unused keys at exit
    time.
    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If the checkpoint was written by the
        name-based `tf.compat.v1.train.Saver`, names are used to match
        variables. This path may also be a SavedModel directory.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.
      The returned status object has the following methods:
      * `assert_consumed()`:
          Raises an exception if any variables are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with other assertions.
      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for
          inference is being loaded. This method returns the status object, and
          so may be chained with other assertions.
      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.
      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).
    Raises:
      NotFoundError: if the a checkpoint or SavedModel cannot be found at
        `save_path`.
    """
    if options and options.experimental_enable_async_checkpoint:
      self._checkpoint_options = options
    # Triggers TF2 async checkpoint handling if:
    # 1. async checkpoint is enabled in CheckpointOptions
    # 2. there's a preceeding async save/write
    # 3. running in eager mode
    if (self._checkpoint_options and
        self._checkpoint_options.experimental_enable_async_checkpoint):
      if context.executing_eagerly():
        return self._async_checkpointer().restore(save_path, options)
      else:
        logging.warning(
            "Saving async checkpoint in graph mode is currently not supported;"
            " switching to regular sync checkpoint instead.")
    orig_save_path = save_path
    if isinstance(save_path, os.PathLike):
      save_path = os.fspath(save_path)
    if save_path is not None and gfile.IsDirectory(save_path) and (
        (gfile.Exists(path_helpers.get_saved_model_pb_path(save_path)) or
         gfile.Exists(path_helpers.get_saved_model_pbtxt_path(save_path)))):
      save_path = path_helpers.get_variables_path(save_path)
    try:
      status = self.read(save_path, options=options)
      if context.executing_eagerly():
        context.async_wait()  # Ensure restore operations have completed.
    except errors_impl.NotFoundError as e:
      raise errors_impl.NotFoundError(
          None, None,
          f"Error when restoring from checkpoint or SavedModel at "
          f"{orig_save_path}: {e.message}"
          f"\nPlease double-check that the path is correct. You may be missing "
          "the checkpoint suffix (e.g. the '-1' in 'path/to/ckpt-1').")
    # Create the save counter now so it gets initialized with other variables
    # when graph building. Creating it earlier would lead to errors when using,
    # say, train.Saver() to save the model before initializing it.
    self._maybe_create_save_counter()
    if isinstance(status, NameBasedSaverStatus):
      status.add_to_optionally_restored(self.save_counter)
    return status
