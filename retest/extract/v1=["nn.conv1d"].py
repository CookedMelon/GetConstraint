@tf_export(v1=["nn.conv1d"])
@dispatch.add_dispatch_support
@deprecation.deprecated_arg_values(
    None,
    "`NCHW` for data_format is deprecated, use `NCW` instead",
    warn_once=True,
    data_format="NCHW")
