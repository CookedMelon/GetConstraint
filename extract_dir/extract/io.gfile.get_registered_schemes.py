@tf_export("io.gfile.get_registered_schemes")
def get_registered_schemes():
  """Returns the currently registered filesystem schemes.
  The `tf.io.gfile` APIs, in addition to accepting traditional filesystem paths,
  also accept file URIs that begin with a scheme. For example, the local
  filesystem path `/tmp/tf` can also be addressed as `file:///tmp/tf`. In this
  case, the scheme is `file`, followed by `://` and then the path, according to
  [URI syntax](https://datatracker.ietf.org/doc/html/rfc3986#section-3).
  This function returns the currently registered schemes that will be recognized
  by `tf.io.gfile` APIs. This includes both built-in schemes and those
  registered by other TensorFlow filesystem implementations, for example those
  provided by [TensorFlow I/O](https://github.com/tensorflow/io).
  The empty string is always included, and represents the "scheme" for regular
  local filesystem paths.
  Returns:
    List of string schemes, e.g. `['', 'file', 'ram']`, in arbitrary order.
  Raises:
    errors.OpError: If the operation fails.
  """
  return _pywrap_file_io.GetRegisteredSchemes()
