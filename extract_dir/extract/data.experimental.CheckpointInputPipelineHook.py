@tf_export("data.experimental.CheckpointInputPipelineHook")
class CheckpointInputPipelineHook(session_run_hook.SessionRunHook):
  """Checkpoints input pipeline state every N steps or seconds.
  This hook saves the state of the iterators in the `Graph` so that when
  training is resumed the input pipeline continues from where it left off.
  This could potentially avoid overfitting in certain pipelines where the
  number of training steps per eval are small compared to the dataset
  size or if the training pipeline is pre-empted.
  Differences from `CheckpointSaverHook`:
  1. Saves only the input pipelines in the "iterators" collection and not the
     global variables or other saveable objects.
  2. Does not write the `GraphDef` and `MetaGraphDef` to the summary.
  Example of checkpointing the training pipeline:
  ```python
  est = tf.estimator.Estimator(model_fn)
  while True:
    est.train(
        train_input_fn,
        hooks=[tf.data.experimental.CheckpointInputPipelineHook(est)],
        steps=train_steps_per_eval)
    # Note: We do not pass the hook here.
    metrics = est.evaluate(eval_input_fn)
    if should_stop_the_training(metrics):
      break
  ```
  This hook should be used if the input pipeline state needs to be saved
  separate from the model checkpoint. Doing so may be useful for a few reasons:
  1. The input pipeline checkpoint may be large, if there are large shuffle
     or prefetch buffers for instance, and may bloat the checkpoint size.
  2. If the input pipeline is shared between training and validation, restoring
     the checkpoint during validation may override the validation input
     pipeline.
  For saving the input pipeline checkpoint alongside the model weights use
  `tf.data.experimental.make_saveable_from_iterator` directly to create a
  `SaveableObject` and add to the `SAVEABLE_OBJECTS` collection. Note, however,
  that you will need to be careful not to restore the training iterator during
  eval. You can do that by not adding the iterator to the SAVEABLE_OBJECTS
  collector when building the eval graph.
  """
  def __init__(self, estimator, external_state_policy=None):
    """Initializes a `CheckpointInputPipelineHook`.
    If the input pipeline depends on external state (e.g. seeds for
    RandomUniform) beyond the input pipeline, this hook would be unable to
    serialize and deserialize that state. If its acceptable to ignore that state
    change the external_state_policy argument to 'warn' or 'ignore'. For e.g.
    ```python
    est = tf.estimator.Estimator(model_fn)
    while True:
      est.train(
          train_input_fn,
          hooks=[tf.data.experimental.CheckpointInputPipelineHook(
              est, external_state_policy='warn')],
          steps=train_steps_per_eval)
      # Note: We do not pass the hook here.
      metrics = est.evaluate(eval_input_fn)
      if should_stop_the_training(metrics):
        break
    ```
    Args:
      estimator: Estimator.
      external_state_policy: A string that identifies how to handle input
        pipelines that depend on external state. Possible values are
        'ignore': The external state is silently ignored.
        'warn': The external state is ignored, logging a warning.
        'fail': The operation fails upon encountering external state.
        By default we set it to 'fail'.
    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of saver or scaffold should be set.
      ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or
        'fail'.
    """
    if external_state_policy is None:
      external_state_policy = "fail"
    self._external_state_policy = _convert_external_state_policy_to_enum(
        external_state_policy)
    # `checkpoint_basename` is "input.ckpt" for non-distributed pipelines or
    # of the form "input_<task_type>_<task_id>.ckpt" for distributed pipelines.
    # Note: The default `checkpoint_basename` used by `CheckpointSaverHook` is
    # "model.ckpt". We intentionally choose the input pipeline checkpoint prefix
    # to be different to avoid conflicts with the model checkpoint.
    # pylint: disable=protected-access
    checkpoint_prefix = "input"
    if estimator._config.num_worker_replicas > 1:
      # Distributed setting.
      suffix = "_{}_{}".format(estimator._config.task_type,
                               estimator._config.task_id)
      checkpoint_prefix += suffix
    # pylint: enable=protected-access
    # We use a composition paradigm instead of inheriting from
    # `CheckpointSaverHook` because `Estimator` does an `isinstance` check
    # to check whether a `CheckpointSaverHook` is already present in the list
    # of hooks and if not, adds one. Inheriting from `CheckpointSaverHook`
    # would thwart this behavior. This hook checkpoints *only the iterators*
    # and not the graph variables.
    self._checkpoint_saver_hook = basic_session_run_hooks.CheckpointSaverHook(
        estimator.model_dir,
        save_secs=estimator._config.save_checkpoints_secs,  # pylint: disable=protected-access
        save_steps=estimator._config.save_checkpoints_steps,  # pylint: disable=protected-access
        checkpoint_basename=checkpoint_prefix + ".ckpt")
    # Name for the protocol buffer file that will contain the list of most
    # recent checkpoints stored as a `CheckpointState` protocol buffer.
    # This file, kept in the same directory as the checkpoint files, is
    # automatically managed by the `Saver` to keep track of recent checkpoints.
    # The default name used by the `Saver` for this file is "checkpoint". Here
    # we use the name "checkpoint_<checkpoint_prefix>" so that in case the
    # `checkpoint_dir` is the same as the model checkpoint directory, there are
    # no conflicts during restore.
    self._latest_filename = "checkpoint_" + checkpoint_prefix
  def begin(self):
    # Build a Saver that saves all iterators in the `GLOBAL_ITERATORS`
    # collection if no `Saver` or `Scaffold` is provided.
    # pylint: disable=protected-access
    if (self._checkpoint_saver_hook._saver is None and
        self._checkpoint_saver_hook._scaffold is None):
      iterators = ops.get_collection(iterator_ops.GLOBAL_ITERATORS)
      saveables = [
          iterator_ops._IteratorSaveable(
              i, i.name, external_state_policy=self._external_state_policy)
          for i in iterators
      ]
      self._checkpoint_saver_hook._saver = _CustomSaver(
          saveables, self._latest_filename, sharded=True)
    # pylint: enable=protected-access
    self._checkpoint_saver_hook.begin()
  def after_create_session(self, session, coord):
    # If a new session was created, we set _first_run to True so that we can
    # restore if needed.
    self._first_run = True
  def _restore_or_save_initial_ckpt(self, session):
    # Ideally this should be run in after_create_session but is not for the
    # following reason:
    # Currently there is no way of enforcing an order of running the
    # `SessionRunHooks`. Hence it is possible that the `_DatasetInitializerHook`
    # is run *after* this hook. That is troublesome because
    # 1. If a checkpoint exists and this hook restores it, the initializer hook
    #    will override it.
    # 2. If no checkpoint exists, this hook will try to save an uninitialized
    #    iterator which will result in an exception.
    #
    # As a temporary fix we enter the following implicit contract between this
    # hook and the _DatasetInitializerHook.
    # 1. The _DatasetInitializerHook initializes the iterator in the call to
    #    after_create_session.
    # 2. This hook saves the iterator on the first call to `before_run()`, which
    #    is guaranteed to happen after `after_create_session()` of all hooks
    #    have been run.
    # Check if there is an existing checkpoint. If so, restore from it.
    # pylint: disable=protected-access
    latest_checkpoint_path = checkpoint_management.latest_checkpoint(
        self._checkpoint_saver_hook._checkpoint_dir,
        latest_filename=self._latest_filename)
    if latest_checkpoint_path:
      self._checkpoint_saver_hook._get_saver().restore(session,
                                                       latest_checkpoint_path)
    else:
      # The checkpoint saved here is the state at step "global_step".
      # Note: We do not save the GraphDef or MetaGraphDef here.
      global_step = session.run(self._checkpoint_saver_hook._global_step_tensor)
      self._checkpoint_saver_hook._save(session, global_step)
      self._checkpoint_saver_hook._timer.update_last_triggered_step(global_step)
    # pylint: enable=protected-access
  def before_run(self, run_context):
    if self._first_run:
      self._restore_or_save_initial_ckpt(run_context.session)
      self._first_run = False
    return self._checkpoint_saver_hook.before_run(run_context)
  def after_run(self, run_context, run_values):
    self._checkpoint_saver_hook.after_run(run_context, run_values)
  def end(self, session):
    self._checkpoint_saver_hook.end(session)
