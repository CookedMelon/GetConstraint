import tensorflow as tf
import os
import tf2onnx

# 定义模型
class SimpleModel(tf.Module):
    def __init__(self):
        self.W1 = tf.Variable(tf.random.normal([2, 10]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([10]), dtype=tf.float32)
        self.W2 = tf.Variable(tf.random.normal([10, 1]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, X):
        layer1 = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        output = tf.sigmoid(tf.matmul(layer1, self.W2) + self.b2)
        return output

# 创建模型实例
my_model = SimpleModel()

# 保存为 SavedModel 格式
model_save_path = "./saved_model"
tf.saved_model.save(my_model, model_save_path)
