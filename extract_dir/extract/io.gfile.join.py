@tf_export("io.gfile.join")
def join(path, *paths):
  r"""Join one or more path components intelligently.
  TensorFlow specific filesystems will be joined
  like a url (using "/" as the path seperator) on all platforms:
  On Windows or Linux/Unix-like:
  >>> tf.io.gfile.join("gcs://folder", "file.py")
  'gcs://folder/file.py'
  >>> tf.io.gfile.join("ram://folder", "file.py")
  'ram://folder/file.py'
  But the native filesystem is handled just like os.path.join:
  >>> path = tf.io.gfile.join("folder", "file.py")
  >>> if os.name == "nt":
  ...   expected = "folder\\file.py"  # Windows
  ... else:
  ...   expected = "folder/file.py"  # Linux/Unix-like
  >>> path == expected
  True
  Args:
    path: string, path to a directory
    paths: string, additional paths to concatenate
  Returns:
    path: the joined path.
  """
  # os.path.join won't take mixed bytes/str, so don't overwrite the incoming `path` var
  path_ = compat.as_str_any(compat.path_to_str(path))
  if "://" in path_[1:]:
    return urljoin(path, *paths)
  return os.path.join(path, *paths)
