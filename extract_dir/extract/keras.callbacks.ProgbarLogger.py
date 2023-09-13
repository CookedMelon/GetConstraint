@keras_export("keras.callbacks.ProgbarLogger")
class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.
    Args:
        count_mode: One of `"steps"` or `"samples"`.
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
            If not provided, defaults to the `Model`'s metrics.
    Raises:
        ValueError: In case of invalid `count_mode`.
    """
    def __init__(self, count_mode: str = "samples", stateful_metrics=None):
        super().__init__()
        self._supports_tf_logs = True
        if count_mode == "samples":
            self.use_steps = False
        elif count_mode == "steps":
            self.use_steps = True
        else:
            raise ValueError(
                f"Unknown `count_mode`: {count_mode}. "
                'Expected values are ["samples", "steps"]'
            )
        # Defaults to all Model's metrics except for loss.
        self.stateful_metrics = (
            set(stateful_metrics) if stateful_metrics else set()
        )
        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1
        self._train_step, self._test_step, self._predict_step = None, None, None
        self._call_batch_hooks = True
        self._called_in_fit = False
    def set_params(self, params):
        self.verbose = params["verbose"]
        self.epochs = params["epochs"]
        if self.use_steps and "steps" in params:
            self.target = params["steps"]
        elif not self.use_steps and "samples" in params:
            self.target = params["samples"]
        else:
            self.target = (
                None  # Will be inferred at the end of the first epoch.
            )
        self._call_batch_hooks = self.verbose == 1
        if self.target is None:
            try:
                self._train_step = self.model._train_counter
                self._test_step = self.model._test_counter
                self._predict_step = self.model._predict_counter
            except AttributeError:
                self._call_batch_hooks = True
    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, validation is silent.
        self._called_in_fit = True
    def on_test_begin(self, logs=None):
        if not self._called_in_fit:
            self._reset_progbar()
            self._maybe_init_progbar()
    def on_predict_begin(self, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()
    def on_epoch_begin(self, epoch, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()
        if self.verbose and self.epochs > 1:
            io_utils.print_msg(f"Epoch {epoch + 1}/{self.epochs}")
    def on_train_batch_end(self, batch, logs=None):
        self._batch_update_progbar(batch, logs)
    def on_test_batch_end(self, batch, logs=None):
        if not self._called_in_fit:
            self._batch_update_progbar(batch, logs)
    def on_predict_batch_end(self, batch, logs=None):
        # Don't pass prediction results.
        self._batch_update_progbar(batch, None)
    def on_epoch_end(self, epoch, logs=None):
        self._finalize_progbar(logs, self._train_step)
    def on_test_end(self, logs=None):
        if not self._called_in_fit:
            self._finalize_progbar(logs, self._test_step)
    def on_predict_end(self, logs=None):
        self._finalize_progbar(logs, self._predict_step)
    def _reset_progbar(self):
        self.seen = 0
        self.progbar = None
    def _maybe_init_progbar(self):
        """Instantiate a `Progbar` if not yet, and update the stateful
        metrics."""
        # TODO(rchao): Legacy TF1 code path may use list for
        # `self.stateful_metrics`. Remove "cast to set" when TF1 support is
        # dropped.
        self.stateful_metrics = set(self.stateful_metrics)
        if self.model:
            # Update the existing stateful metrics as `self.model.metrics` may
            # contain updated metrics after `MetricsContainer` is built in the
            # first train step.
            self.stateful_metrics = self.stateful_metrics.union(
                set(m.name for m in self.model.metrics)
            )
        if self.progbar is None:
            self.progbar = Progbar(
                target=self.target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
                unit_name="step" if self.use_steps else "sample",
            )
        self.progbar._update_stateful_metrics(self.stateful_metrics)
    def _implements_train_batch_hooks(self):
        return self._call_batch_hooks
    def _implements_test_batch_hooks(self):
        return self._call_batch_hooks
    def _implements_predict_batch_hooks(self):
        return self._call_batch_hooks
    def _batch_update_progbar(self, batch, logs=None):
        """Updates the progbar."""
        logs = logs or {}
        self._maybe_init_progbar()
        if self.use_steps:
            self.seen = batch + 1  # One-indexed.
        else:
            # v1 path only.
            logs = copy.copy(logs)
            batch_size = logs.pop("size", 0)
            num_steps = logs.pop("num_steps", 1)
            logs.pop("batch", None)
            add_seen = num_steps * batch_size
            self.seen += add_seen
        if self.verbose == 1:
            # Only block async when verbose = 1.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.progbar.update(self.seen, list(logs.items()), finalize=False)
    def _finalize_progbar(self, logs, counter):
        logs = tf_utils.sync_to_numpy_or_python_type(logs or {})
        if self.target is None:
            if counter is not None:
                counter = counter.numpy()
                if not self.use_steps:
                    counter *= logs.get("size", 1)
            self.target = counter or self.seen
            self.progbar.target = self.target
        self.progbar.update(self.target, list(logs.items()), finalize=True)
