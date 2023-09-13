@tf_export("gather", v1=[])
@dispatch.add_dispatch_support
def gather_v2(params,
              indices,
              validate_indices=None,
              axis=None,
              batch_dims=0,
              name=None):
  return gather(
      params,
      indices,
      validate_indices=validate_indices,
      name=name,
      axis=axis,
      batch_dims=batch_dims)
