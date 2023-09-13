"/home/cc/Workspace/tfconstraint/python/lib/io/tf_record.py"
@tf_export(
    v1=["io.TFRecordCompressionType", "python_io.TFRecordCompressionType"])
@deprecation.deprecated_endpoints("io.TFRecordCompressionType",
                                  "python_io.TFRecordCompressionType")
class TFRecordCompressionType(object):
  """The type of compression for the record."""
  NONE = 0
  ZLIB = 1
  GZIP = 2
