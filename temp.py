import tensorflow as tf
input_tensor = tf.constant([[[
    [[1,2]], [[1,2]], [[1,2]], [[1,2]],
    [[1,2]], [[1,2]], [[1,2]], [[1,2]],
    [[1,2]], [[1,2]], [[1,2]], [[1,2]],
    [[1,2]], [[1,2]], [[1,2]], [[1,2]]
]]], dtype=tf.float32)
print(input_tensor.shape)