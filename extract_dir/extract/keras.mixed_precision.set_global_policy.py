@keras_export("keras.mixed_precision.set_global_policy", v1=[])
def set_global_policy(policy):
    """Sets the global dtype policy.
    The global policy is the default `tf.keras.mixed_precision.Policy` used for
    layers, if no policy is passed to the layer constructor.
    >>> tf.keras.mixed_precision.set_global_policy('mixed_float16')
    >>> tf.keras.mixed_precision.global_policy()
    <Policy "mixed_float16">
    >>> tf.keras.layers.Dense(10).dtype_policy
    <Policy "mixed_float16">
    >>> # Global policy is not used if a policy
    >>> # is directly passed to constructor
    >>> tf.keras.layers.Dense(10, dtype='float64').dtype_policy
    <Policy "float64">
    >>> tf.keras.mixed_precision.set_global_policy('float32')
    If no global policy is set, layers will instead default to a Policy
    constructed from `tf.keras.backend.floatx()`.
    To use mixed precision, the global policy should be set to `'mixed_float16'`
    or `'mixed_bfloat16'`, so that every layer uses a 16-bit compute dtype and
    float32 variable dtype by default.
    Only floating point policies can be set as the global policy, such as
    `'float32'` and `'mixed_float16'`. Non-floating point policies such as
    `'int32'` and `'complex64'` cannot be set as the global policy because most
    layers do not support such policies.
    See `tf.keras.mixed_precision.Policy` for more information.
    Args:
      policy: A Policy, or a string that will be converted to a Policy. Can also
        be None, in which case the global policy will be constructed from
        `tf.keras.backend.floatx()`
    """
    global _global_policy
    if not base_layer_utils.v2_dtype_behavior_enabled():
        raise ValueError(
            "The global policy can only be set in TensorFlow 2 or if "
            "V2 dtype behavior has been set. To enable V2 dtype "
            "behavior, call "
            '"tf.compat.v1.keras.layers.enable_v2_dtype_behavior()"'
        )
    if policy is not None and not isinstance(policy, Policy):
        policy = Policy(policy)
    is_mixed_policy = (
        policy is not None and policy.compute_dtype != policy.variable_dtype
    )
    if is_mixed_policy:
        _check_if_mixed_precision_graph_rewrite_is_enabled(policy)
    if (
        policy is not None
        and policy.compute_dtype is not None
        and not tf.as_dtype(policy.compute_dtype).is_floating
    ):
        raise ValueError(
            "set_global_policy can only be used to set the global "
            'policy to floating-point policies, such as "float32" and '
            f'"mixed_float16", but got policy: {policy.name}'
        )
    _global_policy = policy
    tf.__internal__.train.set_using_mixed_precision_policy(is_mixed_policy)
