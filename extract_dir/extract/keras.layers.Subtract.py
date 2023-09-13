@keras_export("keras.layers.Subtract")
class Subtract(_Merge):
    """Layer that subtracts two inputs.
    It takes as input a list of tensors of size 2, both of the same shape, and
    returns a single tensor, (inputs[0] - inputs[1]), also of the same shape.
    Examples:
    ```python
        import keras
        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])
        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: input_shape={input_shape}"
            )
    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: inputs={inputs}"
            )
        return inputs[0] - inputs[1]
