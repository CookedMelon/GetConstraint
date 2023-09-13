@tf_export("nn.space_to_depth", v1=[])
@dispatch.add_dispatch_support
def space_to_depth_v2(input, block_size, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  return gen_array_ops.space_to_depth(input, block_size, data_format, name=name)
