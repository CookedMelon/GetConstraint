@tf_export(v1=["nn.dropout"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "Please use `rate` instead of `keep_prob`. "
                             "Rate should be set to `rate = 1 - keep_prob`.",
                             "keep_prob")
