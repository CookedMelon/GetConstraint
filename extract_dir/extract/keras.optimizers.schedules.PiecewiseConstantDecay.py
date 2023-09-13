@keras_export("keras.optimizers.schedules.PiecewiseConstantDecay")
class PiecewiseConstantDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a piecewise constant decay schedule.
    The function returns a 1-arg callable to compute the piecewise constant
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.
    Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
      for the next 10000 steps, and 0.1 for any additional steps.
    ```python
    step = tf.Variable(0, trainable=False)
    boundaries = [100000, 110000]
    values = [1.0, 0.5, 0.1]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as the boundary tensors.
      The output of the 1-arg function that takes the `step`
      is `values[0]` when `step <= boundaries[0]`,
      `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
      and values[-1] when `step > boundaries[-1]`.
    """
    def __init__(self, boundaries, values, name=None):
        """Piecewise constant from boundaries and interval values.
        Args:
          boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
            increasing entries, and with all elements having the same type as
            the optimizer step.
          values: A list of `Tensor`s or `float`s or `int`s that specifies the
            values for the intervals defined by `boundaries`. It should have one
            more element than `boundaries`, and all elements should have the
            same type.
          name: A string. Optional name of the operation. Defaults to
            'PiecewiseConstant'.
        Raises:
          ValueError: if the number of elements in the lists do not match.
        """
        super().__init__()
        if len(boundaries) != len(values) - 1:
            raise ValueError(
                "The length of boundaries should be 1 less than the length of "
                f"values. Received: boundaries={boundaries} of length "
                f"{len(boundaries)}, and values={values} "
                f"of length {len(values)}."
            )
        self.boundaries = boundaries
        self.values = values
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstant"):
            boundaries = tf.nest.map_structure(
                tf.convert_to_tensor, tf.nest.flatten(self.boundaries)
            )
            values = tf.nest.map_structure(
                tf.convert_to_tensor, tf.nest.flatten(self.values)
            )
            x_recomp = tf.convert_to_tensor(step)
            for i, b in enumerate(boundaries):
                if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
                    # We cast the boundaries to have the same type as the step
                    b = tf.cast(b, x_recomp.dtype.base_dtype)
                    boundaries[i] = b
            pred_fn_pairs = []
            pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
            pred_fn_pairs.append(
                (x_recomp > boundaries[-1], lambda: values[-1])
            )
            for low, high, v in zip(
                boundaries[:-1], boundaries[1:], values[1:-1]
            ):
                # Need to bind v here; can do this with lambda v=v: ...
                pred = (x_recomp > low) & (x_recomp <= high)
                pred_fn_pairs.append((pred, lambda v=v: v))
            # The default isn't needed here because our conditions are mutually
            # exclusive and exhaustive, but tf.case requires it.
            default = lambda: values[0]
            return tf.case(pred_fn_pairs, default, exclusive=True)
    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "values": self.values,
            "name": self.name,
        }
