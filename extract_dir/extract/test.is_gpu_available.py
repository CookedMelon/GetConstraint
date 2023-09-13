@tf_export("test.is_gpu_available")
def is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
  """Returns whether TensorFlow can access a GPU.
  Warning: if a non-GPU version of the package is installed, the function would
  also return False. Use `tf.test.is_built_with_cuda` to validate if TensorFlow
  was build with CUDA support.
  For example,
  >>> gpu_available = tf.test.is_gpu_available()
  >>> is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
  >>> is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))
  Args:
    cuda_only: limit the search to CUDA GPUs.
    min_cuda_compute_capability: a (major,minor) pair that indicates the minimum
      CUDA compute capability required, or None if no requirement.
  Note that the keyword arg name "cuda_only" is misleading (since routine will
  return true when a GPU device is available irrespective of whether TF was
  built with CUDA support or ROCm support. However no changes here because
  ++ Changing the name "cuda_only" to something more generic would break
     backward compatibility
  ++ Adding an equivalent "rocm_only" would require the implementation check
     the build type. This in turn would require doing the same for CUDA and thus
     potentially break backward compatibility
  ++ Adding a new "cuda_or_rocm_only" would not break backward compatibility,
     but would require most (if not all) callers to update the call to use
     "cuda_or_rocm_only" instead of "cuda_only"
  Returns:
    True if a GPU device of the requested kind is available.
  """
  # This was needed earlier when we had support for SYCL in TensorFlow.
  del cuda_only
  try:
    for local_device in device_lib.list_local_devices():
      if local_device.device_type == "GPU":
        gpu_info = gpu_util.compute_capability_from_device_desc(local_device)
        cc = gpu_info.compute_capability or (0, 0)
        if not min_cuda_compute_capability or cc >= min_cuda_compute_capability:
          return True
    return False
  except errors_impl.NotFoundError as e:
    if not all(x in str(e) for x in ["CUDA", "not find"]):
      raise e
    else:
      logging.error(str(e))
      return False
