@tf_export("io.read_file", v1=["io.read_file", "read_file"])
def read_file(filename, name=None):
  """Reads the contents of file.
  This operation returns a tensor with the entire contents of the input
  filename. It does not do any parsing, it just returns the contents as
  they are. Usually, this is the first step in the input pipeline.
  Example:
  >>> with open("/tmp/file.txt", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.read_file("/tmp/file.txt")
  <tf.Tensor: shape=(), dtype=string, numpy=b'asdf'>
  Example of using the op in a function to read an image, decode it and reshape
  the tensor containing the pixel data:
  >>> @tf.function
  ... def load_image(filename):
  ...   raw = tf.io.read_file(filename)
  ...   image = tf.image.decode_png(raw, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   image.set_shape([28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image
  Args:
    filename: string. filename to read from.
    name: string.  Optional name for the op.
  Returns:
    A tensor of dtype "string", with the file contents.
  """
  return gen_io_ops.read_file(filename, name)
