@tf_export("foldr", v1=[])
@dispatch.add_dispatch_support
@deprecation.deprecated_arg_values(
    None,
    """back_prop=False is deprecated. Consider using tf.stop_gradient instead.
