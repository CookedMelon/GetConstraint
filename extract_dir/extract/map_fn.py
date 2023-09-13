@tf_export("map_fn", v1=[])
@deprecation.deprecated_arg_values(
    None,
    """back_prop=False is deprecated. Consider using tf.stop_gradient instead.
