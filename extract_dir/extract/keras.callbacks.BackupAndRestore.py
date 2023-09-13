@keras_export("keras.callbacks.BackupAndRestore", v1=[])
class BackupAndRestore(Callback):
    """Callback to back up and restore the training state.
    `BackupAndRestore` callback is intended to recover training from an
    interruption that has happened in the middle of a `Model.fit` execution, by
    backing up the training states in a temporary checkpoint file (with the help
    of a `tf.train.CheckpointManager`), at the end of each epoch. Each backup
    overwrites the previously written checkpoint file, so at any given time
    there is at most one such checkpoint file for backup/restoring purpose.
    If training restarts before completion, the training state (which includes
    the `Model` weights and epoch number) is restored to the most recently saved
    state at the beginning of a new `Model.fit` run. At the completion of a
    `Model.fit` run, the temporary checkpoint file is deleted.
    Note that the user is responsible to bring jobs back after the interruption.
    This callback is important for the backup and restore mechanism for fault
    tolerance purpose, and the model to be restored from a previous checkpoint
    is expected to be the same as the one used to back up. If user changes
    arguments passed to compile or fit, the checkpoint saved for fault tolerance
    can become invalid.
    Note:
    1. This callback is not compatible with eager execution disabled.
    2. A checkpoint is saved at the end of each epoch. After restoring,
    `Model.fit` redoes any partial work during the unfinished epoch in which the
    training got restarted (so the work done before the interruption doesn't
    affect the final model state).
    3. This works for both single worker and multi-worker modes. When
    `Model.fit` is used with `tf.distribute`, it supports
    `tf.distribute.MirroredStrategy`,
    `tf.distribute.MultiWorkerMirroredStrategy`, `tf.distribute.TPUStrategy`,
    and `tf.distribute.experimental.ParameterServerStrategy`.
    Example:
    >>> class InterruptingCallback(tf.keras.callbacks.Callback):
    ...   def on_epoch_begin(self, epoch, logs=None):
    ...     if epoch == 4:
    ...       raise RuntimeError('Interrupting!')
    >>> callback = tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> try:
    ...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
    ...             batch_size=1, callbacks=[callback, InterruptingCallback()],
    ...             verbose=0)
    ... except:
    ...   pass
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> # Only 6 more epochs are run, since first training got interrupted at
    >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
    >>> len(history.history['loss'])
    6
    Besides the option to save at the end of every epoch or every N steps, if
    you are doing distributed training with
    `tf.distribute.MultiWorkerMirroredStrategy` on Google Cloud Platform or
    Google Borg, you can also use the `save_before_preemption` argument
    to enable saving a checkpoint right before a worker gets preempted
    by other jobs and training gets interrupted. See
    `tf.distribute.experimental.PreemptionCheckpointHandler` for more details.
    Args:
        backup_dir: String, path to store the checkpoint.
          e.g. `backup_dir = os.path.join(working_dir, 'backup')`.
          This is the directory in which the system stores temporary files to
          recover the model from jobs terminated unexpectedly. The directory
          cannot be reused elsewhere to store other files, e.g. by the
          `BackupAndRestore` callback of another training run,
          or by another callback
          (e.g. `ModelCheckpoint`) of the same training.
        save_freq: `'epoch'`, integer, or `False`. When set to `'epoch'`
          the callback saves the checkpoint at the end of each epoch.
          When set to an integer, the callback saves the checkpoint every
          `save_freq` batches. Set `save_freq` to `False` if only using
          preemption checkpointing (with `save_before_preemption=True`).
        delete_checkpoint: Boolean, default to True. This `BackupAndRestore`
          callback works by saving a checkpoint to back up the training state.
          If `delete_checkpoint=True`, the checkpoint will be deleted after
          training is finished. Use `False` if you'd like to keep the checkpoint
          for future usage.
        save_before_preemption: A boolean value instructing whether to turn on
          the automatic checkpoint saving for preemption/maintenance events.
          This only supports
          `tf.distribute.MultiWorkerMirroredStrategy` on Google Cloud Platform
          or Google Borg for now.
    """
    def __init__(
        self,
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False,
    ):
        super().__init__()
        self.backup_dir = backup_dir
        self._supports_tf_logs = True
        self._supported_strategies = (
            tf.distribute.MirroredStrategy,
            tf.distribute.MultiWorkerMirroredStrategy,
            tf.distribute.experimental.TPUStrategy,
            tf.distribute.TPUStrategy,
            tf.distribute.experimental.ParameterServerStrategy,
        )
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint
        self.save_before_preemption = save_before_preemption
        self._batches_count = 0
        self._current_epoch = 0
        if not tf.executing_eagerly():
            if tf.inside_function():
                raise ValueError(
                    "This Callback's method contains Python state and "
                    "should be called outside of `tf.function`s."
                )
            else:  # Legacy graph mode:
                raise ValueError(
                    "BackupAndRestore only supports eager mode. In graph "
                    "mode, consider using ModelCheckpoint to manually save "
                    "and restore weights with `model.load_weights()` and by "
                    "providing `initial_epoch` in `model.fit()` for fault "
                    "tolerance."
                )
        if (not save_freq) and (not save_before_preemption):
            raise ValueError(
                "Either `save_freq` or `save_before_preemption` " "must be set."
            )
        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False
    def on_train_begin(self, logs=None):
        # TrainingState is used to manage the training state needed for
        # failure-recovery of a worker in training.
        if self.model._distribution_strategy and not isinstance(
            self.model.distribute_strategy, self._supported_strategies
        ):
            raise NotImplementedError(
                f"{type(self.model.distribute_strategy)} is not supported yet. "
                "Currently BackupAndRestore callback "
                "only supports empty strategy, "
                "MirroredStrategy, MultiWorkerMirroredStrategy and TPUStrategy."
            )
        self.model._training_state = worker_training_state.WorkerTrainingState(
            self.model,
            self.backup_dir,
            self.save_freq,
            self.save_before_preemption,
        )
        self._training_state = self.model._training_state
        self._training_state.restore()
    def on_train_batch_begin(self, batch, logs=None):
        # Skip batch update for PSS Strategy
        if isinstance(
            self.model.distribute_strategy,
            tf.distribute.ParameterServerStrategy,
        ):
            return
        self._training_state._ckpt_saved_batch.assign(batch)
    def on_train_batch_end(self, batch, logs=None):
        # Skip batch update for PSS Strategy
        if isinstance(
            self.model.distribute_strategy,
            tf.distribute.ParameterServerStrategy,
        ):
            return
        self._training_state.backup_if_preempted()
        if self.save_freq and self.save_freq != "epoch":
            self._batches_count += 1
            if self._batches_count >= self.save_freq:
                self._batches_count = 0
                self._backup(epoch=self._current_epoch, batch=batch)
    def _implements_train_batch_hooks(self):
        return self.save_freq != "epoch"
    def on_train_end(self, logs=None):
        if self.delete_checkpoint:
            # On exit of training, delete the training state backup file saved
            # for the purpose of worker recovery unless the user opts out.
            self._training_state.delete_backup()
        # Clean up the training state.
        del self._training_state
        del self.model._training_state
    def on_epoch_begin(self, epoch, logs=None):
        self._training_state._ckpt_saved_epoch.assign(epoch)
        self._current_epoch = epoch
    def on_epoch_end(self, epoch, logs=None):
        # Back up the model and current epoch for possible future recovery.
        if self.save_freq == "epoch":
            self._backup(epoch=epoch)
    def _backup(self, epoch, batch=0):
        self._training_state.back_up(epoch=epoch, batch=batch)
