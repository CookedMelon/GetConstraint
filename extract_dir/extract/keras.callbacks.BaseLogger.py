@keras_export("keras.callbacks.BaseLogger")
class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    Args:
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """
    def __init__(self, stateful_metrics=None):
        super().__init__()
        self.stateful_metrics = set(stateful_metrics or [])
    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get("size", 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen`
        # calculation.
        num_steps = logs.get("num_steps", 1)
        self.seen += batch_size * num_steps
        for k, v in logs.items():
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params["metrics"]:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen
