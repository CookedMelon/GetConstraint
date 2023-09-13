@keras_export("keras.callbacks.TerminateOnNaN")
class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            loss = tf_utils.sync_to_numpy_or_python_type(loss)
            if np.isnan(loss) or np.isinf(loss):
                io_utils.print_msg(
                    f"Batch {batch}: Invalid loss, terminating training"
                )
                self.model.stop_training = True
