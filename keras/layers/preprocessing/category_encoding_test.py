# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras text category_encoding preprocessing layer."""


import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import backend
from keras.layers import core
from keras.layers.preprocessing import category_encoding
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class CategoryEncodingInputTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    @parameterized.named_parameters(
        ("list", list),
        ("tuple", tuple),
        ("numpy", np.array),
        ("array_like", preprocessing_test_utils.ArrayLike),
    )
    def test_tensor_like_inputs(self, data_fn):
        category_data = data_fn([1, 2, 3, 3, 0])
        weight_data = data_fn([1, 2, 3, 1, 7])
        expected_output = [7, 1, 2, 4, 0, 0]

        layer = category_encoding.CategoryEncoding(
            num_tokens=6, output_mode=category_encoding.COUNT
        )
        output_data = layer(category_data, count_weights=weight_data)
        self.assertAllEqual(output_data, expected_output)

    def test_compute_output_shape(self):
        layer = category_encoding.CategoryEncoding(5)
        output_shape = layer.compute_output_shape((None, 1))
        self.assertListEqual(output_shape.as_list(), [None, 5])
        output_shape = layer.compute_output_shape([None, 1])
        self.assertListEqual(output_shape.as_list(), [None, 5])

    def test_dense_input_sparse_output(self):
        input_array = tf.constant([[1, 2, 3], [3, 3, 0]])

        # The expected output should be (X for missing value):
        # [[X, 1, 1, 1, X, X]
        #  [1, X, X, 2, X, X]]
        expected_indices = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 3]]
        expected_values = [1, 1, 1, 1, 2]
        num_tokens = 6

        input_data = keras.Input(shape=(None,), dtype=tf.int32)
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=True,
        )
        int_data = layer(input_data)

        model = keras.Model(inputs=input_data, outputs=int_data)
        sp_output_dataset = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_values, sp_output_dataset.values)
        self.assertAllEqual(expected_indices, sp_output_dataset.indices)

        # Assert sparse output is same as dense output.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=False,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array, steps=1)
        self.assertAllEqual(
            tf.sparse.to_dense(sp_output_dataset, default_value=0),
            output_dataset,
        )

    def test_sparse_input(self):
        input_array = np.array([[1, 2, 3, 0], [0, 3, 1, 0]], dtype=np.int64)
        sparse_tensor_data = tf.sparse.from_dense(input_array)

        # pyformat: disable
        expected_output = [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]]
        # pyformat: enable
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        int_data = layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(sparse_tensor_data, steps=1)
        self.assertAllEqual(expected_output, output_dataset)

    def test_sparse_input_with_weights(self):
        input_array = np.array([[1, 2, 3, 4], [4, 3, 1, 4]], dtype=np.int64)
        weights_array = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])
        sparse_tensor_data = tf.sparse.from_dense(input_array)
        sparse_weight_data = tf.sparse.from_dense(weights_array)

        # pyformat: disable
        expected_output = [[0, 0.1, 0.2, 0.3, 0.4, 0], [0, 0.4, 0, 0.1, 0.5, 0]]
        # pyformat: enable
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)
        weight_data = keras.Input(shape=(None,), dtype=tf.float32, sparse=True)

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.COUNT
        )
        int_data = layer(input_data, count_weights=weight_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

        model = keras.Model(inputs=[input_data, weight_data], outputs=int_data)
        output_dataset = model.predict(
            [sparse_tensor_data, sparse_weight_data], steps=1
        )
        self.assertAllClose(expected_output, output_dataset)

    def test_sparse_input_sparse_output(self):
        sp_inp = tf.SparseTensor(
            indices=[[0, 0], [1, 1], [2, 0], [2, 1], [3, 1]],
            values=[0, 2, 1, 1, 0],
            dense_shape=[4, 2],
        )
        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)

        # The expected output should be (X for missing value):
        # [[1, X, X, X]
        #  [X, X, 1, X]
        #  [X, 2, X, X]
        #  [1, X, X, X]]
        expected_indices = [[0, 0], [1, 2], [2, 1], [3, 0]]
        expected_values = [1, 1, 2, 1]
        num_tokens = 6

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=True,
        )
        int_data = layer(input_data)

        model = keras.Model(inputs=input_data, outputs=int_data)
        sp_output_dataset = model.predict(sp_inp, steps=1)
        self.assertAllEqual(expected_values, sp_output_dataset.values)
        self.assertAllEqual(expected_indices, sp_output_dataset.indices)

        # Assert sparse output is same as dense output.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=False,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(sp_inp, steps=1)
        self.assertAllEqual(
            tf.sparse.to_dense(sp_output_dataset, default_value=0),
            output_dataset,
        )

    def test_sparse_input_sparse_output_with_weights(self):
        indices = [[0, 0], [1, 1], [2, 0], [2, 1], [3, 1]]
        sp_inp = tf.SparseTensor(
            indices=indices, values=[0, 2, 1, 1, 0], dense_shape=[4, 2]
        )
        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)
        sp_weight = tf.SparseTensor(
            indices=indices,
            values=[0.1, 0.2, 0.4, 0.3, 0.2],
            dense_shape=[4, 2],
        )
        weight_data = keras.Input(shape=(None,), dtype=tf.float32, sparse=True)

        # The expected output should be (X for missing value):
        # [[1, X, X, X]
        #  [X, X, 1, X]
        #  [X, 2, X, X]
        #  [1, X, X, X]]
        expected_indices = [[0, 0], [1, 2], [2, 1], [3, 0]]
        expected_values = [0.1, 0.2, 0.7, 0.2]
        num_tokens = 6

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=True,
        )
        int_data = layer(input_data, count_weights=weight_data)

        model = keras.Model(inputs=[input_data, weight_data], outputs=int_data)
        sp_output_dataset = model.predict([sp_inp, sp_weight], steps=1)
        self.assertAllClose(expected_values, sp_output_dataset.values)
        self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    def test_ragged_input(self):
        input_array = tf.ragged.constant([[1, 2, 3], [3, 1]])

        # pyformat: disable
        expected_output = [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]]
        # pyformat: enable
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        input_data = keras.Input(shape=(None,), dtype=tf.int32, ragged=True)

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        int_data = layer(input_data)

        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_output, output_dataset)

    def test_ragged_input_sparse_output(self):
        input_array = tf.ragged.constant([[1, 2, 3], [3, 3]])

        # The expected output should be (X for missing value):
        # [[X, 1, 1, 1]
        #  [X, X, X, 2]]
        expected_indices = [[0, 1], [0, 2], [0, 3], [1, 3]]
        expected_values = [1, 1, 1, 2]
        num_tokens = 6

        input_data = keras.Input(shape=(None,), dtype=tf.int32, ragged=True)
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=True,
        )
        int_data = layer(input_data)

        model = keras.Model(inputs=input_data, outputs=int_data)
        sp_output_dataset = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_values, sp_output_dataset.values)
        self.assertAllEqual(expected_indices, sp_output_dataset.indices)

        # Assert sparse output is same as dense output.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=False,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array, steps=1)
        self.assertAllEqual(
            tf.sparse.to_dense(sp_output_dataset, default_value=0),
            output_dataset,
        )

    def test_sparse_output_and_dense_layer(self):
        input_array = tf.constant([[1, 2, 3], [3, 3, 0]])

        num_tokens = 4

        input_data = keras.Input(shape=(None,), dtype=tf.int32)
        encoding_layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=category_encoding.COUNT,
            sparse=True,
        )
        int_data = encoding_layer(input_data)
        dense_layer = keras.layers.Dense(units=1)
        output_data = dense_layer(int_data)

        model = keras.Model(inputs=input_data, outputs=output_data)
        _ = model.predict(input_array, steps=1)

    def test_dense_oov_input(self):
        valid_array = tf.constant([[0, 1, 2], [0, 1, 2]])
        invalid_array = tf.constant([[0, 1, 2], [2, 3, 1]])
        num_tokens = 3
        expected_output_shape = [None, num_tokens]
        encoder_layer = category_encoding.CategoryEncoding(num_tokens)
        input_data = keras.Input(shape=(3,), dtype=tf.int32)
        int_data = encoder_layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())
        model = keras.Model(inputs=input_data, outputs=int_data)
        # Call predict once on valid input to compile a graph and test control
        # flow.
        _ = model.predict(valid_array, steps=1)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            ".*must be in the range 0 <= values < num_tokens.*",
        ):
            _ = model.predict(invalid_array, steps=1)

    def test_dense_negative(self):
        valid_array = tf.constant([[0, 1, 2], [0, 1, 2]])
        invalid_array = tf.constant([[1, 2, 0], [2, 2, -1]])
        num_tokens = 3
        expected_output_shape = [None, num_tokens]
        encoder_layer = category_encoding.CategoryEncoding(num_tokens)
        input_data = keras.Input(shape=(3,), dtype=tf.int32)
        int_data = encoder_layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())
        model = keras.Model(inputs=input_data, outputs=int_data)
        # Call predict once on valid input to compile a graph and test control
        # flow.
        _ = model.predict(valid_array, steps=1)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            ".*must be in the range 0 <= values < num_tokens.*",
        ):
            _ = model.predict(invalid_array, steps=1)

    def test_legacy_max_tokens_arg(self):
        input_array = np.array([[1, 2, 3, 1]])
        expected_output = [[0, 1, 1, 1, 0, 0]]
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        input_data = keras.Input(shape=(None,), dtype=tf.int32)
        layer = category_encoding.CategoryEncoding(
            max_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        int_data = layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)


@test_combinations.run_all_keras_modes
class CategoryEncodingOutputTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    @parameterized.named_parameters(
        ("float32", tf.float32),
        ("float64", tf.float64),
    )
    def test_output_dtype(self, dtype):
        inputs = keras.Input(shape=(1,), dtype=tf.int32)
        layer = category_encoding.CategoryEncoding(
            num_tokens=4, output_mode=category_encoding.ONE_HOT, dtype=dtype
        )
        outputs = layer(inputs)
        self.assertAllEqual(outputs.dtype, dtype)

    def test_one_hot_output(self):
        input_data = np.array([[3], [2], [0], [1]])
        expected_output = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        num_tokens = 4
        expected_output_shape = [None, num_tokens]

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT
        )
        inputs = keras.Input(shape=(1,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_dataset = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_dataset)

    def test_one_hot_output_rank_one_input(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        num_tokens = 4
        expected_output_shape = [None, num_tokens]

        # Test call on layer directly.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = keras.Input(shape=(1,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_output_rank_zero_input(self):
        input_data = np.array(3)
        expected_output = [0, 0, 0, 1]
        num_tokens = 4
        expected_output_shape = [None, num_tokens]

        # Test call on layer directly.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = keras.Input(shape=(1,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)

        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_rank_3_output_fails(self):
        layer = category_encoding.CategoryEncoding(
            num_tokens=4, output_mode=category_encoding.ONE_HOT
        )
        with self.assertRaisesRegex(
            ValueError, "maximum supported output rank"
        ):
            _ = layer(keras.Input(shape=(4,), dtype=tf.int32))
        with self.assertRaisesRegex(
            ValueError, "maximum supported output rank"
        ):
            _ = layer(np.array([[3, 2, 0, 1], [3, 2, 0, 1]]))

    def test_multi_hot_output(self):
        input_data = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        expected_output = [
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
        ]
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        inputs = keras.Input(shape=(None,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model.predict(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_output_rank_one_input(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = [1, 1, 1, 1, 0, 0]
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        # Test call on layer directly.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = keras.Input(shape=(4,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_output_rank_zero_input(self):
        input_data = np.array(3)
        expected_output = [0, 0, 0, 1, 0, 0]
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        # Test call on layer directly.
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = keras.Input(shape=(4,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_rank_3_output_fails(self):
        layer = category_encoding.CategoryEncoding(
            num_tokens=4, output_mode=category_encoding.ONE_HOT
        )
        with self.assertRaisesRegex(
            ValueError, "maximum supported output rank"
        ):
            _ = layer(
                keras.Input(
                    shape=(
                        3,
                        4,
                    ),
                    dtype=tf.int32,
                )
            )
        with self.assertRaisesRegex(
            ValueError, "maximum supported output rank"
        ):
            _ = layer(np.array([[[3, 2, 0, 1], [3, 2, 0, 1]]]))

    def test_count_output(self):
        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

        # pyformat: disable
        expected_output = [[0, 2, 1, 1, 0, 0], [2, 1, 0, 1, 0, 0]]
        # pyformat: enable
        num_tokens = 6
        expected_output_shape = [None, num_tokens]

        input_data = keras.Input(shape=(None,), dtype=tf.int32)
        layer = category_encoding.CategoryEncoding(
            num_tokens=6, output_mode=category_encoding.COUNT
        )
        int_data = layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)


class CategoryEncodingModelBuildingTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    @parameterized.named_parameters(
        {
            "testcase_name": "count_output",
            "num_tokens": 5,
            "output_mode": category_encoding.COUNT,
        },
        {
            "testcase_name": "multi_hot_output",
            "num_tokens": 5,
            "output_mode": category_encoding.MULTI_HOT,
        },
    )
    def test_end_to_end_bagged_modeling(self, output_mode, num_tokens):
        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

        input_data = keras.Input(shape=(None,), dtype=tf.int32)
        layer = category_encoding.CategoryEncoding(
            num_tokens=num_tokens, output_mode=output_mode
        )

        weights = []
        if num_tokens is None:
            layer.set_num_elements(5)
        layer.set_weights(weights)

        int_data = layer(input_data)
        float_data = backend.cast(int_data, dtype="float32")
        output_data = core.Dense(64)(float_data)
        model = keras.Model(inputs=input_data, outputs=output_data)
        _ = model.predict(input_array)


if __name__ == "__main__":
    tf.test.main()
