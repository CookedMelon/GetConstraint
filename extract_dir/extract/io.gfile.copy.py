@tf_export("io.gfile.copy")
def copy_v2(src, dst, overwrite=False):
  """Copies data from `src` to `dst`.
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("/tmp/x")
  True
  >>> tf.io.gfile.copy("/tmp/x", "/tmp/y")
  >>> tf.io.gfile.exists("/tmp/y")
  True
  >>> tf.io.gfile.remove("/tmp/y")
  You can also specify the URI scheme for selecting a different filesystem:
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
  >>> tf.io.gfile.exists("/tmp/y")
  True
  >>> tf.io.gfile.remove("/tmp/y")
  Note that you need to always specify a file name, even if moving into a new
  directory. This is because some cloud filesystems don't have the concept of a
  directory.
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.mkdir("/tmp/new_dir")
  >>> tf.io.gfile.copy("/tmp/x", "/tmp/new_dir/y")
  >>> tf.io.gfile.exists("/tmp/new_dir/y")
  True
  >>> tf.io.gfile.rmtree("/tmp/new_dir")
  If you want to prevent errors if the path already exists, you can use
  `overwrite` argument:
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y", overwrite=True)
  >>> tf.io.gfile.remove("/tmp/y")
  Note that the above will still result in an error if you try to overwrite a
  directory with a file.
  Note that you cannot copy a directory, only file arguments are supported.
  Args:
    src: string, name of the file whose contents need to be copied
    dst: string, name of the file to which to copy to
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.
  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.CopyFile(
      compat.path_to_bytes(src), compat.path_to_bytes(dst), overwrite)
