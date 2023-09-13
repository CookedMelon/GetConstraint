# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras Premade Linear models."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras import losses
from keras.engine import input_layer
from keras.engine import sequential
from keras.engine import training
from keras.feature_column import dense_features_v2
from keras.layers import core
from keras.optimizers.legacy import gradient_descent
from keras.premade_models import linear
from keras.testing_infra import test_combinations


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class LinearModelTest(test_combinations.TestCase):
    def test_linear_model_with_single_input(self):
        model = linear.LinearModel()
        inp = np.random.uniform(low=-5.0, high=5.0, size=(64, 2))
        output = 0.3 * inp[:, 0] + 0.2 * inp[:, 1]
        model.compile("sgd", "mse", [])
        model.fit(inp, output, epochs=5)
        self.assertTrue(model.built)

    def test_linear_model_with_list_input(self):
        model = linear.LinearModel()
        input_a = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        input_b = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        output = 0.3 * input_a + 0.2 * input_b
        model.compile("sgd", "mse", [])
        model.fit([input_a, input_b], output, epochs=5)

    def test_linear_model_with_mismatched_dict_inputs(self):
        model = linear.LinearModel()
        input_a = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        input_b = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        output = 0.3 * input_a + 0.2 * input_b
        model.compile("sgd", "mse", [])
        model.build(
            {"a": tf.TensorShape([None, 1]), "b": tf.TensorShape([None, 1])}
        )
        with self.assertRaisesRegex(ValueError, "Missing keys"):
            model.fit({"c": input_a, "b": input_b}, output, epochs=5)

    def test_linear_model_with_dict_input(self):
        model = linear.LinearModel()
        input_a = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        input_b = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        output = 0.3 * input_a + 0.2 * input_b
        model.compile("sgd", "mse", [])
        model.fit({"a": input_a, "b": input_b}, output, epochs=5)

    def test_linear_model_as_layer(self):
        input_a = input_layer.Input(shape=(1,), name="a")
        output_a = linear.LinearModel()(input_a)
        input_b = input_layer.Input(shape=(1,), name="b")
        output_b = core.Dense(units=1)(input_b)
        output = output_a + output_b
        model = training.Model(inputs=[input_a, input_b], outputs=[output])
        input_a_np = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        input_b_np = np.random.uniform(low=-5.0, high=5.0, size=(64, 1))
        output_np = 0.3 * input_a_np + 0.2 * input_b_np
        model.compile("sgd", "mse", [])
        model.fit([input_a_np, input_b_np], output_np, epochs=5)

    def test_linear_model_with_sparse_input(self):
        indices = tf.constant([[0, 0], [0, 2], [1, 0], [1, 1]], dtype=tf.int64)
        values = tf.constant([0.4, 0.6, 0.8, 0.5])
        shape = tf.constant([2, 3], dtype=tf.int64)
        model = linear.LinearModel()
        inp = tf.SparseTensor(indices, values, shape)
        output = model(inp)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        if tf.executing_eagerly():
            weights = model.get_weights()
            weights[0] = np.ones((3, 1))
            model.set_weights(weights)
            output = model(inp)
            self.assertAllClose([[1.0], [1.3]], self.evaluate(output))

    def test_linear_model_with_sparse_input_and_custom_training(self):
        batch_size = 64
        indices = []
        values = []
        target = np.zeros((batch_size, 1))
        for i in range(64):
            rand_int = np.random.randint(3)
            if rand_int == 0:
                indices.append((i, 0))
                val = np.random.uniform(low=-5.0, high=5.0)
                values.append(val)
                target[i] = 0.3 * val
            elif rand_int == 1:
                indices.append((i, 1))
                val = np.random.uniform(low=-5.0, high=5.0)
                values.append(val)
                target[i] = 0.2 * val
            else:
                indices.append((i, 0))
                indices.append((i, 1))
                val_1 = np.random.uniform(low=-5.0, high=5.0)
                val_2 = np.random.uniform(low=-5.0, high=5.0)
                values.append(val_1)
                values.append(val_2)
                target[i] = 0.3 * val_1 + 0.2 * val_2

        indices = np.asarray(indices)
        values = np.asarray(values)
        shape = tf.constant([batch_size, 2], dtype=tf.int64)
        inp = tf.SparseTensor(indices, values, shape)
        model = linear.LinearModel(use_bias=False)
        opt = gradient_descent.SGD()
        for _ in range(20):
            with tf.GradientTape() as t:
                output = model(inp)
                loss = backend.mean(losses.mean_squared_error(target, output))
            grads = t.gradient(loss, model.trainable_variables)
            grads_and_vars = zip(grads, model.trainable_variables)
            opt.apply_gradients(grads_and_vars)

    # This test is an example for a regression on categorical inputs, i.e.,
    # the output is 0.4, 0.6, 0.9 when input is 'alpha', 'beta', 'gamma'
    # separately.
    def test_linear_model_with_feature_column(self):
        vocab_list = ["alpha", "beta", "gamma"]
        vocab_val = [0.4, 0.6, 0.9]
        data = np.random.choice(vocab_list, size=256)
        y = np.zeros_like(data, dtype=np.float32)
        for vocab, val in zip(vocab_list, vocab_val):
            indices = np.where(data == vocab)
            y[indices] = val + np.random.uniform(
                low=-0.01, high=0.01, size=indices[0].shape
            )
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key="symbol", vocabulary_list=vocab_list
        )
        ind_column = tf.feature_column.indicator_column(cat_column)
        dense_feature_layer = dense_features_v2.DenseFeatures([ind_column])
        linear_model = linear.LinearModel(
            use_bias=False, kernel_initializer="zeros"
        )
        combined = sequential.Sequential([dense_feature_layer, linear_model])
        opt = gradient_descent.SGD(learning_rate=0.1)
        combined.compile(opt, "mse", [])
        combined.fit(x={"symbol": data}, y=y, batch_size=32, epochs=10)
        self.assertAllClose(
            [[0.4], [0.6], [0.9]],
            combined.layers[1].dense_layers[0].kernel.numpy(),
            atol=0.01,
        )

    def test_config(self):
        linear_model = linear.LinearModel(units=3, use_bias=True)
        config = linear_model.get_config()
        cloned_linear_model = linear.LinearModel.from_config(config)
        self.assertEqual(linear_model.units, cloned_linear_model.units)


if __name__ == "__main__":
    tf.test.main()
