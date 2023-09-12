import tensorflow as tf

# 定义模型参数
W1 = tf.Variable(tf.random.normal([2, 10]), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([10]), dtype=tf.float32)

W2 = tf.Variable(tf.random.normal([10, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)
print(b1,b2)
# 定义模型
def model(X):
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    output = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    return output

# 使用模型进行预测（假设输入）
sample_input = tf.constant([[0.5, 0.5]], dtype=tf.float32)
sample_output = model(sample_input)

print("Sample output:", sample_output.numpy())

multiple_samples = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=tf.float32)  # Shape: [3, 2]
output = model(multiple_samples)
print("Sample output:", output.numpy())
