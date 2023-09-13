@tf_export("nn.RNNCellResidualWrapper", v1=[])
class ResidualWrapper(_RNNCellWrapper):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""
    def __init__(self, cell, residual_fn=None, **kwargs):
        """Constructs a `ResidualWrapper` for `cell`.
        Args:
          cell: An instance of `RNNCell`.
          residual_fn: (Optional) The function to map raw cell inputs and raw
            cell outputs to the actual cell outputs of the residual network.
            Defaults to calling nest.map_structure on (lambda i, o: i + o),
            inputs and outputs.
          **kwargs: dict of keyword arguments for base layer.
        """
        super().__init__(cell, **kwargs)
        self._residual_fn = residual_fn
    def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
        """Run the cell and apply the residual_fn.
        Args:
          inputs: cell inputs.
          state: cell state.
          cell_call_fn: Wrapped cell's method to use for step computation
            (cell's `__call__` or 'call' method).
          **kwargs: Additional arguments passed to the wrapped cell's `call`.
        Returns:
          Tuple of cell outputs and new state.
        Raises:
          TypeError: If cell inputs and outputs have different structure (type).
          ValueError: If cell inputs and outputs have different structure
            (value).
        """
        outputs, new_state = cell_call_fn(inputs, state, **kwargs)
        # Ensure shapes match
        def assert_shape_match(inp, out):
            inp.get_shape().assert_is_compatible_with(out.get_shape())
        def default_residual_fn(inputs, outputs):
            tf.nest.assert_same_structure(inputs, outputs)
            tf.nest.map_structure(assert_shape_match, inputs, outputs)
            return tf.nest.map_structure(
                lambda inp, out: inp + out, inputs, outputs
            )
        res_outputs = (self._residual_fn or default_residual_fn)(
            inputs, outputs
        )
        return (res_outputs, new_state)
    def get_config(self):
        """Returns the config of the residual wrapper."""
        if self._residual_fn is not None:
            (
                function,
                function_type,
                function_module,
            ) = _serialize_function_to_config(self._residual_fn)
            config = {
                "residual_fn": function,
                "residual_fn_type": function_type,
                "residual_fn_module": function_module,
            }
        else:
            config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "residual_fn" in config:
            config = config.copy()
            residual_function = _parse_config_to_function(
                config,
                custom_objects,
                "residual_fn",
                "residual_fn_type",
                "residual_fn_module",
            )
            config["residual_fn"] = residual_function
        return super(ResidualWrapper, cls).from_config(
            config, custom_objects=custom_objects
        )
