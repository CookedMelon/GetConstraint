@tf_export("io.gfile.exists")
def file_exists_v2(path):
  """Determines whether a path exists or not.
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("/tmp/x")
  True
  You can also specify the URI scheme for selecting a different filesystem:
  >>> # for a GCS filesystem path:
  >>> # tf.io.gfile.exists("gs://bucket/file")
  >>> # for a local filesystem:
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("file:///tmp/x")
  True
  This currently returns `True` for existing directories but don't rely on this
  behavior, especially if you are using cloud filesystems (e.g., GCS, S3,
  Hadoop):
  >>> tf.io.gfile.exists("/tmp")
  True
  Args:
    path: string, a path
  Returns:
    True if the path exists, whether it's a file or a directory.
    False if the path does not exist and there are no filesystem errors.
  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  """
  try:
    _pywrap_file_io.FileExists(compat.path_to_bytes(path))
  except errors.NotFoundError:
    return False
  return True
