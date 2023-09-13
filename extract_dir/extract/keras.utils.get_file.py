@keras_export("keras.utils.get_file")
def get_file(
    fname=None,
    origin=None,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
