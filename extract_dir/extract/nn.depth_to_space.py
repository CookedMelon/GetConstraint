@tf_export("nn.depth_to_space", v1=[])
@dispatch.add_dispatch_support
def depth_to_space_v2(input, block_size, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  return gen_array_ops.depth_to_space(input, block_size, data_format, name=name)
