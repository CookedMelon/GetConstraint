@keras_export("keras.applications.vgg16.VGG16", "keras.applications.VGG16")
def VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
