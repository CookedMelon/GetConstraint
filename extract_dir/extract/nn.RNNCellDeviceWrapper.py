@tf_export("nn.RNNCellDeviceWrapper", v1=[])
class DeviceWrapper(_RNNCellWrapper):
    """Operator that ensures an RNNCell runs on a particular device."""
    def __init__(self, cell, device, **kwargs):
        """Construct a `DeviceWrapper` for `cell` with device `device`.
        Ensures the wrapped `cell` is called with `tf.device(device)`.
        Args:
          cell: An instance of `RNNCell`.
          device: A device string or function, for passing to `tf.device`.
          **kwargs: dict of keyword arguments for base layer.
        """
        super().__init__(cell, **kwargs)
        self._device = device
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState"):
            with tf.compat.v1.device(self._device):
                return self.cell.zero_state(batch_size, dtype)
    def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
        """Run the cell on specified device."""
        with tf.compat.v1.device(self._device):
            return cell_call_fn(inputs, state, **kwargs)
    def get_config(self):
        config = {"device": self._device}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
