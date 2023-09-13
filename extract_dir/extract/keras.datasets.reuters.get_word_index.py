@keras_export("keras.datasets.reuters.get_word_index")
def get_word_index(path="reuters_word_index.json"):
    """Retrieves a dict mapping words to their index in the Reuters dataset.
    Actual word indices starts from 3, with 3 indices reserved for:
    0 (padding), 1 (start), 2 (oov).
    E.g. word index of 'the' is 1, but the in the actual training data, the
    index of 'the' will be 1 + 3 = 4. Vice versa, to translate word indices in
    training data back to words using this mapping, indices need to substract 3.
    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).
    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    """
    origin_folder = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "reuters_word_index.json",
        file_hash="4d44cc38712099c9e383dc6e5f11a921",
    )
    with open(path) as f:
        return json.load(f)
