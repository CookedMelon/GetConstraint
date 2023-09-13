@tf_export("nn.separable_conv2d", v1=[])
@dispatch.add_dispatch_support
def separable_conv2d_v2(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
    name=None,
