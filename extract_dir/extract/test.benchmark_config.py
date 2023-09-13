@tf_export("test.benchmark_config")
def benchmark_config():
  """Returns a tf.compat.v1.ConfigProto for disabling the dependency optimizer.
    Returns:
      A TensorFlow ConfigProto object.
  """
  config = config_pb2.ConfigProto()
  config.graph_options.rewrite_options.dependency_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return config
