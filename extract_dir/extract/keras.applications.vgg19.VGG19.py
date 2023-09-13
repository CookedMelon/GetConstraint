@keras_export("keras.applications.vgg19.VGG19", "keras.applications.VGG19")
def VGG19(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
