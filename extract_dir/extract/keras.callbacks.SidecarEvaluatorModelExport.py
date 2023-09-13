@keras_export("keras.callbacks.SidecarEvaluatorModelExport")
class SidecarEvaluatorModelExport(ModelCheckpoint):
    """Callback to save the best Keras model.
    It expands the functionality of the existing ModelCheckpoint callback to
    enable exporting the best models after evaluation with validation dataset.
    When using the `SidecarEvaluatorModelExport` callback in conjunction with
    `keras.utils.SidecarEvaluator`, users should provide the `filepath`, which
    is the path for this callback to export model or save weights to, and
    `ckpt_filepath`, which is where the checkpoint is available to extract
    the epoch number from. The callback will then export the model that the
    evaluator deems as the best (among the checkpoints saved by the training
    counterpart) to the specified `filepath`. This callback is intended to be
    used by SidecarEvaluator only.
    Example:
    ```python
    model.compile(loss=..., optimizer=...,
                  metrics=['accuracy'])
    sidecar_evaluator = keras.utils.SidecarEvaluator(
        model=model,
        data=dataset,
        checkpoint_dir=checkpoint_dir,
        max_evaluations=1,
        callbacks=[
            SidecarEvaluatorModelExport(
                export_filepath=os.path.join(checkpoint_dir,
                                      'best_model_eval',
                                      'best-model-{epoch:04d}'),
                checkpoint_filepath=os.path.join(checkpoint_dir,
                'ckpt-{epoch:04d}'),
                save_freq="eval",
                save_weights_only=True,
                monitor="loss",
                mode="min",
                verbose=1,
            ),
        ],
    )
    sidecar_evaluator.start()
    # Model weights are saved if evaluator deems it's the best seen so far.
    Args:
        export_filepath: Path where best models should be saved by this
          `SidecarEvaluatorModelExport` callback. Epoch formatting options, such
          as `os.path.join(best_model_dir, 'best-model-{epoch:04d}')`, can be
          used to allow saved model to preserve epoch information in the file
          name. SidecarEvaluatorModelExport will use the "training epoch" at
          which the checkpoint was saved by training to fill the epoch
          placeholder in the path.
        checkpoint_filepath: Path where checkpoints were saved by training. This
          should be the same as what is provided to `filepath` argument of
          `ModelCheckpoint` on the training side, such as
          `os.path.join(checkpoint_dir, 'ckpt-{epoch:04d}')`.
    """
    def __init__(self, export_filepath, checkpoint_filepath, **kwargs):
        super().__init__(
            filepath=export_filepath,
            save_best_only=True,
            **kwargs,
        )
        self._checkpoint_filepath = checkpoint_filepath
    def on_test_begin(self, logs=None):
        """Updates export_index to the latest checkpoint."""
        most_recent_filepath = (
            self._get_most_recently_modified_file_matching_pattern(
                self._checkpoint_filepath
            )
        )
        if most_recent_filepath is not None:
            self.export_index = (
                int(
                    re.match(r".*ckpt-(?P<ckpt>\d+)", most_recent_filepath)[
                        "ckpt"
                    ]
                )
                - 1
            )
        else:
            self.export_index = 0
    def on_test_end(self, logs):
        """Saves best model at the end of an evaluation epoch."""
        self.epochs_since_last_save += 1
        self._save_model(epoch=self.export_index, batch=None, logs=logs)
