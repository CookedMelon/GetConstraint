@keras_export("keras.preprocessing.text.hashing_trick")
def hashing_trick(
    text,
    n,
    hash_function=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    analyzer=None,
