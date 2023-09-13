@tf_export("gather_nd", v1=[])
@dispatch.add_dispatch_support
def gather_nd_v2(params, indices, batch_dims=0, name=None):
  return gather_nd(params, indices, name=name, batch_dims=batch_dims)
