@tf_export("space_to_batch", "nn.space_to_batch", v1=[])
@dispatch.add_dispatch_support
def space_to_batch_v2(input, block_shape, paddings, name=None):  # pylint: disable=redefined-builtin
  return space_to_batch_nd(input, block_shape, paddings, name)
