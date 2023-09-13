@keras_export("keras.datasets.imdb.get_word_index")
def get_word_index(path="imdb_word_index.json"):
    """Retrieves a dict mapping words to their index in the IMDB dataset.
    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).
    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    Example:
    ```python
    # Use the default parameters to keras.datasets.imdb.load_data
    start_char = 1
    oov_char = 2
    index_from = 3
    # Retrieve the training sequences.
    (x_train, _), _ = keras.datasets.imdb.load_data(
        start_char=start_char, oov_char=oov_char, index_from=index_from
    )
    # Retrieve the word index file mapping words to indices
    word_index = keras.datasets.imdb.get_word_index()
    # Reverse the word index to obtain a dict mapping indices to words
    # And add `index_from` to indices to sync with `x_train`
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )
    # Update `inverted_word_index` to include `start_char` and `oov_char`
    inverted_word_index[start_char] = "[START]"
    inverted_word_index[oov_char] = "[OOV]"
    # Decode the first sequence in the dataset
    decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
    ```
    """
    origin_folder = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "imdb_word_index.json",
        file_hash="bfafd718b763782e994055a2d397834f",
    )
    with open(path) as f:
        return json.load(f)
