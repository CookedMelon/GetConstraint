@tf_export("nn.crelu", v1=[])
@dispatch.add_dispatch_support
def crelu_v2(features, axis=-1, name=None):
  return crelu(features, name=name, axis=axis)
