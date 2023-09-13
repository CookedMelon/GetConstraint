@keras_export("keras.layers.GlobalMaxPooling1D", "keras.layers.GlobalMaxPool1D")
class GlobalMaxPooling1D(GlobalPooling1D):
    """Global max pooling operation for 1D temporal data.
    Downsamples the input representation by taking the maximum value over
    the time dimension.
    For example:
    >>> x = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> x = tf.reshape(x, [3, 3, 1])
    >>> x
    <tf.Tensor: shape=(3, 3, 1), dtype=float32, numpy=
    array([[[1.], [2.], [3.]],
           [[4.], [5.], [6.]],
           [[7.], [8.], [9.]]], dtype=float32)>
    >>> max_pool_1d = tf.keras.layers.GlobalMaxPooling1D()
    >>> max_pool_1d(x)
    <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    array([[3.],
           [6.],
           [9.], dtype=float32)>
    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the temporal dimension are retained with
        length 1.
        The behavior is the same as for `tf.reduce_max` or `np.max`.
    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`
    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    """
    def call(self, inputs):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        return backend.max(inputs, axis=steps_axis, keepdims=self.keepdims)
