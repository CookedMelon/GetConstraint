"/home/cc/Workspace/tfconstraint/keras/constraints.py"
@keras_export(
    "keras.constraints.RadialConstraint", "keras.constraints.radial_constraint"
)
class RadialConstraint(Constraint):
    """Constrains `Conv2D` kernel weights to be the same for each radius.
    Also available via the shortcut function
    `tf.keras.constraints.radial_constraint`.
    For example, the desired output for the following 4-by-4 kernel:
    ```
        kernel = [[v_00, v_01, v_02, v_03],
                  [v_10, v_11, v_12, v_13],
                  [v_20, v_21, v_22, v_23],
                  [v_30, v_31, v_32, v_33]]
    ```
    is this::
    ```
        kernel = [[v_11, v_11, v_11, v_11],
                  [v_11, v_33, v_33, v_11],
                  [v_11, v_33, v_33, v_11],
                  [v_11, v_11, v_11, v_11]]
    ```
    This constraint can be applied to any `Conv2D` layer version, including
    `Conv2DTranspose` and `SeparableConv2D`, and with either `"channels_last"`
    or `"channels_first"` data format. The method assumes the weight tensor is
    of shape `(rows, cols, input_depth, output_depth)`.
    """
    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        w_shape = w.shape
        if w_shape.rank is None or w_shape.rank != 4:
            raise ValueError(
                "The weight tensor must have rank 4. "
                f"Received weight tensor with shape: {w_shape}"
            )
        height, width, channels, kernels = w_shape
        w = backend.reshape(w, (height, width, channels * kernels))
        # TODO(cpeter): Switch map_fn for a faster tf.vectorized_map once
        # backend.switch is supported.
        w = backend.map_fn(
            self._kernel_constraint,
            backend.stack(tf.unstack(w, axis=-1), axis=0),
        )
        return backend.reshape(
            backend.stack(tf.unstack(w, axis=0), axis=-1),
            (height, width, channels, kernels),
        )
    def _kernel_constraint(self, kernel):
        """Radially constraints a kernel with shape (height, width,
        channels)."""
        padding = backend.constant([[1, 1], [1, 1]], dtype="int32")
        kernel_shape = backend.shape(kernel)[0]
        start = backend.cast(kernel_shape / 2, "int32")
        kernel_new = backend.switch(
            backend.cast(tf.math.floormod(kernel_shape, 2), "bool"),
            lambda: kernel[start - 1 : start, start - 1 : start],
            lambda: kernel[start - 1 : start, start - 1 : start]
            + backend.zeros((2, 2), dtype=kernel.dtype),
        )
        index = backend.switch(
            backend.cast(tf.math.floormod(kernel_shape, 2), "bool"),
            lambda: backend.constant(0, dtype="int32"),
            lambda: backend.constant(1, dtype="int32"),
        )
        while_condition = lambda index, *args: backend.less(index, start)
        def body_fn(i, array):
            return i + 1, tf.pad(
                array, padding, constant_values=kernel[start + i, start + i]
            )
        _, kernel_new = tf.compat.v1.while_loop(
            while_condition,
            body_fn,
            [index, kernel_new],
            shape_invariants=[index.get_shape(), tf.TensorShape([None, None])],
        )
        return kernel_new
# Aliases.
max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm
radial_constraint = RadialConstraint
# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm
