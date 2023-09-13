@keras_export("keras.callbacks.Callback")
class Callback:
    """Abstract base class used to build new callbacks.
    Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
    `predict` in order to hook into the various stages of the model training and
    inference lifecycle.
    To create a custom callback, subclass `keras.callbacks.Callback` and
    override the method associated with the stage of interest. See
    https://www.tensorflow.org/guide/keras/custom_callback for more information.
    Example:
    >>> training_finished = False
    >>> class MyCallback(tf.keras.callbacks.Callback):
    ...   def on_train_end(self, logs=None):
    ...     global training_finished
    ...     training_finished = True
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(1, input_shape=(1,))])
    >>> model.compile(loss='mean_squared_error')
    >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
    ...           callbacks=[MyCallback()])
    >>> assert training_finished == True
    If you want to use `Callback` objects in a custom training loop:
    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop. Like this:
    Example:
    ```python
       callbacks =  tf.keras.callbacks.CallbackList([...])
       callbacks.append(...)
       callbacks.on_train_begin(...)
       for epoch in range(EPOCHS):
         callbacks.on_epoch_begin(epoch)
         for i, data in dataset.enumerate():
           callbacks.on_train_batch_begin(i)
           batch_logs = model.train_step(data)
           callbacks.on_train_batch_end(i, batch_logs)
         epoch_logs = ...
         callbacks.on_epoch_end(epoch, epoch_logs)
       final_logs=...
       callbacks.on_train_end(final_logs)
    ```
    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """
    def __init__(self):
        self.validation_data = None
        self.model = None
        # Whether this Callback should only run on the chief worker in a
        # Multi-Worker setting.
        # TODO(omalleyt): Make this attr public once solution is stable.
        self._chief_worker_only = None
        self._supports_tf_logs = False
    def set_params(self, params):
        self.params = params
    def set_model(self, model):
        self.model = model
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
    @doc_controls.for_subclass_implementers
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.
        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.
        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
    @doc_controls.for_subclass_implementers
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    @doc_controls.for_subclass_implementers
    def on_predict_end(self, logs=None):
        """Called at the end of prediction.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
    def _implements_train_batch_hooks(self):
        """Determines if this Callback should be called for each train batch."""
        return (
            not generic_utils.is_default(self.on_batch_begin)
            or not generic_utils.is_default(self.on_batch_end)
            or not generic_utils.is_default(self.on_train_batch_begin)
            or not generic_utils.is_default(self.on_train_batch_end)
        )
    def _implements_test_batch_hooks(self):
        """Determines if this Callback should be called for each test batch."""
        return not generic_utils.is_default(
            self.on_test_batch_begin
        ) or not generic_utils.is_default(self.on_test_batch_end)
    def _implements_predict_batch_hooks(self):
        """Determines if this Callback should be called for each predict
        batch."""
        return not generic_utils.is_default(
            self.on_predict_batch_begin
        ) or not generic_utils.is_default(self.on_predict_batch_end)
