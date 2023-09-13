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
"""Tests for Keras text vectorization preprocessing layer."""

import itertools
import math
import os
import random
import string

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.layers.preprocessing import index_lookup
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import CustomObjectScope


def zip_and_sort(weight_values):
    keys, values = weight_values
    return sorted(zip(keys, values), key=lambda x: x[1])


def _get_end_to_end_test_cases():
    test_cases = (
        {
            "testcase_name": "test_strings_soft_vocab_cap",
            # Create an array where 'earth' is the most frequent term, followed
            # by 'wind', then 'and', then 'fire'. This ensures that the vocab
            # accumulator is sorting by frequency.
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array(
                [
                    ["earth"],
                    ["wind"],
                    ["and"],
                    ["fire"],
                    ["fire"],
                    ["and"],
                    ["earth"],
                    ["michigan"],
                ]
            ),
            "kwargs": {
                "max_tokens": None,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "vocabulary_dtype": tf.string,
            },
            "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
            "input_dtype": tf.string,
        },
        {
            "testcase_name": "test_inverse_strings_soft_vocab_cap",
            # Create an array where 'earth' is the most frequent term, followed
            # by 'wind', then 'and', then 'fire'. This ensures that the vocab
            # accumulator is sorting by frequency.
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array([[2], [3], [4], [1], [1], [4], [2], [5]]),
            "kwargs": {
                "max_tokens": None,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "vocabulary_dtype": tf.string,
                "invert": True,
            },
            "expected_output": np.array(
                [
                    [b"earth"],
                    [b"wind"],
                    [b"and"],
                    [b"[OOV]"],
                    [b"[OOV]"],
                    [b"and"],
                    [b"earth"],
                    [b"fire"],
                ]
            ),
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_strings_with_special_tokens",
            # Mask and oov values in the vocab data should be dropped, and
            # mapped to 0 and 1 respectively when calling the layer.
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    [""],
                    [""],
                    [""],
                    ["[OOV]"],
                    ["[OOV]"],
                    ["[OOV]"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array(
                [
                    ["earth"],
                    [""],
                    ["wind"],
                    ["[OOV]"],
                    ["and"],
                    [""],
                    ["fire"],
                    ["and"],
                    ["[OOV]"],
                    ["michigan"],
                ]
            ),
            "kwargs": {
                "max_tokens": None,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "vocabulary_dtype": tf.string,
            },
            "expected_output": [
                [2],
                [0],
                [3],
                [1],
                [4],
                [0],
                [5],
                [4],
                [1],
                [1],
            ],
            "input_dtype": tf.string,
        },
        {
            "testcase_name": "test_ints_soft_vocab_cap",
            # Create an array where 1138 is the most frequent term, followed by
            # 1729, then 725, then 42. This ensures that the vocab accumulator
            # is sorting by frequency.
            "vocab_data": np.array(
                [
                    [42],
                    [1138],
                    [1138],
                    [1138],
                    [1138],
                    [1729],
                    [1729],
                    [1729],
                    [725],
                    [725],
                ],
                dtype=np.int64,
            ),
            "input_data": np.array(
                [[1138], [1729], [725], [42], [42], [725], [1138], [4]],
                dtype=np.int64,
            ),
            "kwargs": {
                "max_tokens": None,
                "num_oov_indices": 1,
                "mask_token": 0,
                "oov_token": -1,
                "vocabulary_dtype": tf.int64,
            },
            "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_ints_with_special_tokens",
            # Mask and oov values in the vocab data should be dropped, and
            # mapped to 0 and 1 respectively when calling the layer.
            "vocab_data": np.array(
                [
                    [42],
                    [1138],
                    [1138],
                    [1138],
                    [1138],
                    [0],
                    [0],
                    [0],
                    [-1],
                    [-1],
                    [-1],
                    [1729],
                    [1729],
                    [1729],
                    [725],
                    [725],
                ],
                dtype=np.int64,
            ),
            "input_data": np.array(
                [[1138], [0], [1729], [-1], [725], [0], [42], [725], [-1], [4]],
                dtype=np.int64,
            ),
            "kwargs": {
                "max_tokens": None,
                "num_oov_indices": 1,
                "mask_token": 0,
                "oov_token": -1,
                "vocabulary_dtype": tf.int64,
            },
            "expected_output": [
                [2],
                [0],
                [3],
                [1],
                [4],
                [0],
                [5],
                [4],
                [1],
                [1],
            ],
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_strings_hard_vocab_cap",
            # Create an array where 'earth' is the most frequent term, followed
            # by 'wind', then 'and', then 'fire'. This ensures that the vocab
            # accumulator is sorting by frequency.
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array(
                [
                    ["earth"],
                    ["wind"],
                    ["and"],
                    ["fire"],
                    ["fire"],
                    ["and"],
                    ["earth"],
                    ["michigan"],
                ]
            ),
            "kwargs": {
                "max_tokens": 5,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "vocabulary_dtype": tf.string,
            },
            "expected_output": [[2], [3], [4], [1], [1], [4], [2], [1]],
            "input_dtype": tf.string,
        },
        {
            "testcase_name": "test_inverse_strings_hard_vocab_cap",
            # Create an array where 'earth' is the most frequent term, followed
            # by 'wind', then 'and', then 'fire'. This ensures that the vocab
            # accumulator is sorting by frequency.
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array([[2], [3], [4], [1], [1], [4], [2], [5]]),
            "kwargs": {
                "max_tokens": 5,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "vocabulary_dtype": tf.string,
                "invert": True,
            },
            "expected_output": np.array(
                [
                    [b"earth"],
                    [b"wind"],
                    [b"and"],
                    [b"[OOV]"],
                    [b"[OOV]"],
                    [b"and"],
                    [b"earth"],
                    [b"[OOV]"],
                ]
            ),
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_ints_hard_vocab_cap",
            # Create an array where 1138 is the most frequent term, followed by
            # 1729, then 725, then 42. This ensures that the vocab accumulator
            # is sorting by frequency.
            "vocab_data": np.array(
                [
                    [42],
                    [1138],
                    [1138],
                    [1138],
                    [1138],
                    [1729],
                    [1729],
                    [1729],
                    [725],
                    [725],
                ],
                dtype=np.int64,
            ),
            "input_data": np.array(
                [[1138], [1729], [725], [42], [42], [725], [1138], [4]],
                dtype=np.int64,
            ),
            "kwargs": {
                "max_tokens": 5,
                "num_oov_indices": 1,
                "mask_token": 0,
                "oov_token": -1,
                "vocabulary_dtype": tf.int64,
            },
            "expected_output": [[2], [3], [4], [1], [1], [4], [2], [1]],
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_ints_tf_idf_output",
            "vocab_data": np.array(
                [
                    [42],
                    [1138],
                    [1138],
                    [1138],
                    [1138],
                    [1729],
                    [1729],
                    [1729],
                    [725],
                    [725],
                ]
            ),
            "input_data": np.array(
                [[1138], [1729], [725], [42], [42], [725], [1138], [4]]
            ),
            "kwargs": {
                "max_tokens": 5,
                "pad_to_max_tokens": True,
                "num_oov_indices": 1,
                "mask_token": 0,
                "oov_token": -1,
                "output_mode": index_lookup.TF_IDF,
                "vocabulary_dtype": tf.int64,
            },
            "expected_output": [
                [0, 1.098612, 0, 0, 0],
                [0, 0, 1.252763, 0, 0],
                [0, 0, 0, 1.466337, 0],
                [0, 0, 0, 0, 1.7917595],
                [0, 0, 0, 0, 1.7917595],
                [0, 0, 0, 1.4663371, 0],
                [0, 1.098612, 0, 0, 0],
                [1.402368, 0, 0, 0, 0],
            ],
            "input_dtype": tf.int64,
        },
        {
            "testcase_name": "test_strings_tf_idf_output",
            "vocab_data": np.array(
                [
                    ["fire"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["earth"],
                    ["wind"],
                    ["wind"],
                    ["wind"],
                    ["and"],
                    ["and"],
                ]
            ),
            "input_data": np.array(
                [
                    ["earth"],
                    ["wind"],
                    ["and"],
                    ["fire"],
                    ["fire"],
                    ["and"],
                    ["earth"],
                    ["michigan"],
                ]
            ),
            "kwargs": {
                "max_tokens": 5,
                "pad_to_max_tokens": True,
                "num_oov_indices": 1,
                "mask_token": "",
                "oov_token": "[OOV]",
                "output_mode": index_lookup.TF_IDF,
                "vocabulary_dtype": tf.string,
            },
            "expected_output": [
                [0, 1.098612, 0, 0, 0],
                [0, 0, 1.252763, 0, 0],
                [0, 0, 0, 1.466337, 0],
                [0, 0, 0, 0, 1.7917595],
                [0, 0, 0, 0, 1.7917595],
                [0, 0, 0, 1.4663371, 0],
                [0, 1.098612, 0, 0, 0],
                [1.402368, 0, 0, 0, 0],
            ],
            "input_dtype": tf.string,
        },
    )

    crossed_test_cases = []
    # Cross above test cases with use_dataset in (True, False)
    for use_dataset in (True, False):
        for case in test_cases:
            case = case.copy()
            if use_dataset:
                case["testcase_name"] = case["testcase_name"] + "_with_dataset"
            case["use_dataset"] = use_dataset
            crossed_test_cases.append(case)

    return crossed_test_cases


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupLayerTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    @parameterized.named_parameters(*_get_end_to_end_test_cases())
    def test_layer_end_to_end_with_adapt(
        self,
        vocab_data,
        input_data,
        kwargs,
        use_dataset,
        expected_output,
        input_dtype,
    ):
        cls = index_lookup.IndexLookup
        if "invert" in kwargs and kwargs["invert"]:
            expected_output_dtype = kwargs["vocabulary_dtype"]
        elif (
            "output_mode" in kwargs
            and kwargs["output_mode"] != index_lookup.INT
        ):
            expected_output_dtype = tf.float32
        else:
            expected_output_dtype = tf.int64

        input_shape = input_data.shape

        if use_dataset:
            # Keras APIs expect batched datasets.
            # TODO(rachelim): `model.predict` predicts the result on each
            # dataset batch separately, then tries to concatenate the results
            # together. When the results have different shapes on the non-concat
            # axis (which can happen in the output_mode = INT case for
            # IndexLookup), the concatenation fails. In real use cases, this may
            # not be an issue because users are likely to pipe the preprocessing
            # layer into other keras layers instead of predicting it directly. A
            # workaround for these unit tests is to have the dataset only
            # contain one batch, so no concatenation needs to happen with the
            # result. For consistency with numpy input, we should make `predict`
            # join differently shaped results together sensibly, with 0 padding.
            input_data = tf.data.Dataset.from_tensor_slices(input_data).batch(
                input_shape[0]
            )
            vocab_data = tf.data.Dataset.from_tensor_slices(vocab_data).batch(
                input_shape[0]
            )

        with CustomObjectScope({"IndexLookup": cls}):
            output_data = test_utils.layer_test(
                cls,
                kwargs=kwargs,
                input_shape=input_shape,
                input_data=input_data,
                input_dtype=input_dtype,
                expected_output_dtype=expected_output_dtype,
                validate_training=False,
                adapt_data=vocab_data,
            )
        if "invert" in kwargs and kwargs["invert"]:
            self.assertAllEqual(expected_output, output_data)
        else:
            self.assertAllClose(expected_output, output_data)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class CategoricalEncodingInputTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_sparse_string_input(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=["fire", "michigan"],
            dense_shape=[3, 4],
        )

        expected_indices = [[0, 0], [1, 2]]
        expected_values = [5, 1]
        expected_dense_shape = [3, 4]

        input_data = keras.Input(shape=(None,), dtype=tf.string, sparse=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_indices, output_data.indices)
        self.assertAllEqual(expected_values, output_data.values)
        self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

    def test_sparse_int_input(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=np.array([13, 32], dtype=np.int64),
            dense_shape=[3, 4],
        )

        expected_indices = [[0, 0], [1, 2]]
        expected_values = [5, 1]
        expected_dense_shape = [3, 4]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_indices, output_data.indices)
        self.assertAllEqual(expected_values, output_data.values)
        self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

    def test_ragged_string_input(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.ragged.constant(
            [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]]
        )
        expected_output = [[2, 3, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.string, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_ragged_int_input(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.ragged.constant(
            [[10, 11, 13], [13, 12, 10, 42]], dtype=np.int64
        )
        expected_output = [[2, 3, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int32_input_with_int64_keys(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.ragged.constant(
            [[10, 11, 13], [13, 12, 10, 42]], dtype=np.int32
        )
        expected_output = [[2, 3, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.int32, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class CategoricalEncodingMultiOOVTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_sparse_string_input_multi_bucket(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=["fire", "ohio"],
            dense_shape=[3, 4],
        )

        expected_indices = [[0, 0], [1, 2]]
        expected_values = [6, 2]
        expected_dense_shape = [3, 4]

        input_data = keras.Input(shape=(None,), dtype=tf.string, sparse=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=2,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_indices, output_data.indices)
        self.assertAllEqual(expected_values, output_data.values)
        self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

    def test_sparse_int_input_multi_bucket(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=np.array([13, 133], dtype=np.int64),
            dense_shape=[3, 4],
        )

        expected_indices = [[0, 0], [1, 2]]
        expected_values = [6, 2]
        expected_dense_shape = [3, 4]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=2,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_indices, output_data.indices)
        self.assertAllEqual(expected_values, output_data.values)
        self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

    def test_ragged_string_input_multi_bucket(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.ragged.constant(
            [["earth", "wind", "fire"], ["fire", "and", "earth", "ohio"]]
        )
        expected_output = [[3, 4, 6], [6, 5, 3, 2]]

        input_data = keras.Input(shape=(None,), dtype=tf.string, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=2,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_ragged_int_input_multi_bucket(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.ragged.constant(
            [[10, 11, 13], [13, 12, 10, 133]], dtype=np.int64
        )
        expected_output = [[3, 4, 6], [6, 5, 3, 2]]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=2,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class CategoricalEncodingAdaptTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_sparse_adapt(self):
        vocab_data = tf.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 2]],
            values=["michigan", "fire", "michigan"],
            dense_shape=[3, 4],
        )
        vocab_dataset = tf.data.Dataset.from_tensors(vocab_data)

        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.adapt(vocab_dataset)
        expected_vocabulary = ["", "[OOV]", "michigan", "fire"]
        self.assertAllEqual(expected_vocabulary, layer.get_vocabulary())

    def test_ragged_adapt(self):
        vocab_data = tf.ragged.constant([["michigan"], ["fire", "michigan"]])
        vocab_dataset = tf.data.Dataset.from_tensors(vocab_data)

        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.adapt(vocab_dataset)
        expected_vocabulary = ["", "[OOV]", "michigan", "fire"]
        self.assertAllEqual(expected_vocabulary, layer.get_vocabulary())

    def test_sparse_int_input(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=np.array([13, 32], dtype=np.int64),
            dense_shape=[3, 4],
        )

        expected_indices = [[0, 0], [1, 2]]
        expected_values = [5, 1]
        expected_dense_shape = [3, 4]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, sparse=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(input_array, steps=1)
        self.assertAllEqual(expected_indices, output_data.indices)
        self.assertAllEqual(expected_values, output_data.values)
        self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

    def test_ragged_string_input(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.ragged.constant(
            [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]]
        )
        expected_output = [[2, 3, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.string, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_ragged_int_input(self):
        vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
        input_array = tf.ragged.constant(
            [[10, 11, 13], [13, 12, 10, 42]], dtype=np.int64
        )
        expected_output = [[2, 3, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.int64, ragged=True)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            vocabulary_dtype=tf.int64,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_single_string_generator_dataset(self):
        def word_gen():
            for _ in itertools.count(1):
                yield "".join(
                    random.choice(string.ascii_letters) for i in range(2)
                )

        ds = tf.data.Dataset.from_generator(
            word_gen, tf.string, tf.TensorShape([])
        )
        batched_ds = ds.take(2)
        input_t = keras.Input(shape=(), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=10,
            num_oov_indices=0,
            mask_token=None,
            oov_token=None,
            vocabulary_dtype=tf.string,
        )
        _ = layer(input_t)
        layer.adapt(batched_ds)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupOutputTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def _write_to_temp_file(self, file_name, vocab_list):
        vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
        with tf.io.gfile.GFile(vocab_path, "w") as writer:
            for vocab in vocab_list:
                writer.write(vocab + "\n")
            writer.flush()
            writer.close()
        return vocab_path

    @parameterized.product(
        rank=[0, 1, 2],
        # Check lists, numpy arrays, tensors, and objects convertable to tensor.
        data_fn=[
            None,
            np.array,
            tf.constant,
            preprocessing_test_utils.ArrayLike,
        ],
    )
    def test_input_types(self, rank, data_fn):
        input_data = vocab = ["earth", "wind", "and", "fire"]
        expected_output = [2, 3, 4, 5]
        if rank == 0:
            input_data = input_data[0]
            expected_output = expected_output[0]
        elif rank == 2:
            input_data = [input_data]
            expected_output = [expected_output]
        if data_fn is not None:
            input_data = data_fn(input_data)
        input_shape = [] if rank == 0 else [None]

        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary=vocab,
            vocabulary_dtype=tf.string,
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Again in a keras.Model
        inputs = keras.Input(shape=input_shape, dtype=tf.string)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        output_data = model(tf.constant(input_data))
        self.assertAllEqual(expected_output, output_data)

    def test_int_output_shape(self):
        input_data = keras.Input(batch_size=16, shape=(4,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=2,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        self.assertAllEqual(int_data.shape.as_list(), [16, 4])

    @parameterized.named_parameters(
        ("int32", tf.int32),
        ("int64", tf.int64),
    )
    def test_int_output_dtype(self, dtype):
        input_data = keras.Input(batch_size=16, shape=(4,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=2,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            dtype=dtype,
        )
        int_data = layer(input_data)
        self.assertAllEqual(int_data.dtype, dtype)

    def test_int_output_float_dtype_fails(self):
        with self.assertRaisesRegex(ValueError, "`dtype` should be an integer"):
            index_lookup.IndexLookup(
                max_tokens=2,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                vocabulary_dtype=tf.string,
                dtype=tf.float32,
            )

    def test_int_output_no_reserved_zero(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=None,
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int_output_no_oov(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        valid_input = np.array(
            [["earth", "wind", "and", "fire"], ["fire", "and", "earth", ""]]
        )
        invalid_input = np.array(
            [
                ["earth", "wind", "and", "michigan"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=0,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(valid_input)
        self.assertAllEqual(expected_output, output_data)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "found OOV values.*michigan"
        ):
            _ = model.predict(invalid_input)

    def test_int_output_no_oov_ragged(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        valid_input = np.array(
            [["earth", "wind", "and", "fire"], ["fire", "and", "earth", ""]]
        )
        invalid_input = np.array(
            [
                ["earth", "wind", "and", "michigan"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        valid_input = tf.RaggedTensor.from_tensor(valid_input)
        invalid_input = tf.RaggedTensor.from_tensor(invalid_input)
        expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=0,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(valid_input)
        self.assertAllEqual(expected_output, output_data)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "found OOV values.*michigan"
        ):
            _ = model.predict(invalid_input)

    def test_int_output_no_oov_sparse(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        valid_input = np.array(
            [["earth", "wind", "and", "fire"], ["fire", "and", "earth", ""]]
        )
        invalid_input = np.array(
            [
                ["earth", "wind", "and", "michigan"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        valid_input = tf.sparse.from_dense(valid_input)
        invalid_input = tf.sparse.from_dense(invalid_input)
        expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=0,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_data = model.predict(valid_input)
        self.assertAllEqual(expected_output, tf.sparse.to_dense(output_data))
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "found OOV values.*michigan"
        ):
            _ = model.predict(invalid_input)

    def test_int_output_explicit_vocab(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_one_hot_output_hard_maximum(self):
        """Check binary output when pad_to_max_tokens=True."""
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(["earth", "wind", "and", "fire", "michigan", ""])
        expected_output = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        input_data = keras.Input(shape=(1,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=6,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.ONE_HOT,
            pad_to_max_tokens=True,
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        binary_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=binary_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_one_hot_output_soft_maximum(self):
        """Check binary output when pad_to_max_tokens=False."""
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(["earth", "wind", "and", "fire", "michigan", ""])
        expected_output = [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        input_data = keras.Input(shape=(1,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.ONE_HOT,
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        binary_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=binary_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_one_hot_output_rank_zero_no_oov(self):
        """Check binary output when pad_to_max_tokens=False."""
        vocab_data = ["earth", "wind", "and", "fire"]
        input_data = tf.constant("earth")
        expected_output = [1, 0, 0, 0]

        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=0,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.ONE_HOT,
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_output_shape(self):
        inputs = keras.Input(batch_size=16, shape=(1,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=["earth"],
            max_tokens=2,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.ONE_HOT,
            vocabulary_dtype=tf.string,
        )
        outputs = layer(inputs)
        self.assertAllEqual(outputs.shape.as_list(), [16, 2])

    @parameterized.product(
        sparse=[True, False],
        adapt=[True, False],
        pad_to_max=[True, False],
        mode=["multi_hot", "count", "tf_idf"],
        dtype=[tf.float32, tf.float64],
    )
    def test_binned_output(self, sparse, adapt, pad_to_max, mode, dtype):
        """Check "multi_hot", "count", and "tf_idf" output."""
        # Adapt breaks ties with sort order.
        vocab_data = ["wind", "fire", "earth", "and"]
        # IDF weight for a term in 1 out of 1 document is log(1 + 1/2).
        idf_data = [math.log(1.5)] * 4
        input_data = np.array(
            [
                ["and", "earth", "fire", "and", ""],
                ["michigan", "wind", "and", "ohio", ""],
            ]
        )

        if mode == "count":
            expected_output = np.array(
                [
                    [0, 0, 1, 1, 2],
                    [2, 1, 0, 0, 1],
                ]
            )
        elif mode == "tf_idf":
            expected_output = np.array(
                [
                    [0, 0, 1, 1, 2],
                    [2, 1, 0, 0, 1],
                ]
            ) * math.log(1.5)
        else:
            expected_output = np.array(
                [
                    [0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                ]
            )
        expected_output_shape = [None, 5]
        if pad_to_max:
            expected_output = np.concatenate(
                (expected_output, [[0], [0]]), axis=1
            )
            expected_output_shape = [None, 6]

        inputs = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=6,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            output_mode=mode,
            pad_to_max_tokens=pad_to_max,
            vocabulary_dtype=tf.string,
            sparse=sparse,
            vocabulary=None if adapt else vocab_data,
            idf_weights=None if adapt or mode != "tf_idf" else idf_data,
            dtype=dtype,
        )
        if adapt:
            layer.adapt(vocab_data)
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        output_data = model.predict(input_data)
        if sparse:
            output_data = tf.sparse.to_dense(output_data)
        # Check output data.
        self.assertAllClose(expected_output, output_data)
        # Check symbolic output shape.
        self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
        # Check output dtype.
        self.assertAllEqual(dtype, output_data.dtype)

    def test_multi_hot_output_no_oov(self):
        """Check multi hot output when num_oov_indices=0."""
        vocab_data = ["earth", "wind", "and", "fire"]
        valid_input = np.array(
            [["earth", "wind", "and", "fire"], ["fire", "and", "earth", ""]]
        )
        invalid_input = np.array(
            [
                ["earth", "wind", "and", "michigan"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
        ]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=0,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.MULTI_HOT,
            pad_to_max_tokens=True,
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        binary_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=binary_data)
        output_data = model.predict(valid_input)
        self.assertAllEqual(expected_output, output_data)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "found OOV values.*michigan"
        ):
            _ = model.predict(invalid_input)

    def test_multi_hot_output_hard_maximum_multiple_adapts(self):
        input_array = np.array(
            [
                ["earth", "wind", "and", "earth"],
                ["ohio", "and", "earth", "michigan"],
            ]
        )
        adapt_data = [
            "earth",
            "earth",
            "earth",
            "earth",
            "wind",
            "wind",
            "wind",
        ]
        first_expected_output = [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
        second_adapt_data = [
            "earth",
            "earth",
            "earth",
            "earth",
            "wind",
            "wind",
            "wind",
            "and",
            "and",
            "fire",
        ]
        second_expected_output = [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
        ]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            output_mode=index_lookup.MULTI_HOT,
            pad_to_max_tokens=True,
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)

        # Test the first adapt
        layer.adapt(adapt_data)
        first_output = model.predict(input_array)
        # Test the second adapt
        layer.adapt(second_adapt_data)
        # We need to recompile the model to retrace our call graph.
        model.compile()
        second_output = model.predict(input_array)
        self.assertAllEqual(first_expected_output, first_output)
        self.assertAllEqual(second_expected_output, second_output)

    def test_int_output_file_vocab(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 0, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_non_int_output_file_vocab_in_tf_function(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = tf.constant(
            [
                ["earth", "wind", "and", "fire", ""],
                ["fire", "and", "earth", "michigan", ""],
            ],
            dtype=tf.string,
        )

        expected_output = [
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
        ]
        vocab_file = self._write_to_temp_file("temp", vocab_data)

        @tf.function
        def compute(data):
            layer = index_lookup.IndexLookup(
                vocabulary=vocab_file,
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                output_mode=index_lookup.MULTI_HOT,
                vocabulary_dtype=tf.string,
            )
            return layer(data)

        output_dataset = compute(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_file_vocab_and_list_vocab_identical_attrs(self):
        vocab_data = ["earth", "wind", "and", "fire"]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        file_layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        list_layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        expected_vocab = ["", "[OOV]", "earth", "wind", "and", "fire"]
        self.assertAllEqual(expected_vocab, list_layer.get_vocabulary())
        expected_vocab_size = 6
        self.assertAllEqual(expected_vocab_size, list_layer.vocabulary_size())
        self.assertAllEqual(
            list_layer.get_vocabulary(), file_layer.get_vocabulary()
        )
        self.assertAllEqual(
            list_layer.vocabulary_size(), file_layer.vocabulary_size()
        )

    def test_file_vocab_and_list_vocab_identical_attrs_multi_oov(self):
        vocab_data = ["earth", "wind", "and", "fire"]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        file_layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            num_oov_indices=2,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        list_layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=2,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        expected_vocab = ["", "[OOV]", "[OOV]", "earth", "wind", "and", "fire"]
        self.assertAllEqual(expected_vocab, list_layer.get_vocabulary())
        expected_vocab_size = 7
        self.assertAllEqual(expected_vocab_size, list_layer.vocabulary_size())
        self.assertAllEqual(
            list_layer.get_vocabulary(), file_layer.get_vocabulary()
        )
        self.assertAllEqual(
            list_layer.vocabulary_size(), file_layer.vocabulary_size()
        )

    def test_file_vocab_and_list_vocab_identical_attrs_no_mask(self):
        vocab_data = ["earth", "wind", "and", "fire"]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        file_layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            num_oov_indices=2,
            mask_token=None,
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        list_layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=2,
            mask_token=None,
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )

        expected_vocab = ["[OOV]", "[OOV]", "earth", "wind", "and", "fire"]
        self.assertAllEqual(expected_vocab, list_layer.get_vocabulary())
        expected_vocab_size = 6
        self.assertAllEqual(expected_vocab_size, list_layer.vocabulary_size())
        self.assertAllEqual(
            list_layer.get_vocabulary(), file_layer.get_vocabulary()
        )
        self.assertAllEqual(
            list_layer.vocabulary_size(), file_layer.vocabulary_size()
        )

    def test_int_output_file_vocab_no_mask(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "", "earth", "michigan"],
            ]
        )
        expected_output = [[1, 2, 3, 4], [4, 0, 1, 0]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            mask_token=None,
            num_oov_indices=1,
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int_output_file_vocab_no_oov_or_mask(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [["earth", "wind", "and", "fire"], ["fire", "wind", "earth", "and"]]
        )
        expected_output = [[0, 1, 2, 3], [3, 1, 0, 2]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            mask_token=None,
            num_oov_indices=0,
            oov_token=None,
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int_output_file_vocab_inversion(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array([[1, 2, 3, 4], [4, 0, 1, 0]])
        expected_output = [
            ["earth", "wind", "and", "fire"],
            ["fire", "[OOV]", "earth", "[OOV]"],
        ]

        vocab_file = self._write_to_temp_file("temp", vocab_data)
        idata = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            mask_token=None,
            num_oov_indices=1,
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        _ = layer(idata)

        input_data = keras.Input(shape=(None,), dtype=tf.int64)

        invert_layer = index_lookup.IndexLookup(
            vocabulary=layer.get_vocabulary(),
            max_tokens=None,
            oov_token="[OOV]",
            mask_token=None,
            num_oov_indices=1,
            invert=True,
            vocabulary_dtype=tf.string,
        )
        int_data = invert_layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int_output_int_file_vocab(self):
        vocab_data = ["10", "20", "30", "40"]
        input_array = np.array([[10, 20, 30, 40], [40, 0, 10, 42]])
        expected_output = [[2, 3, 4, 5], [5, 0, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)
        input_data = keras.Input(shape=(None,), dtype=tf.int64)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_file,
            max_tokens=None,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_dataset_map_output(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=0,
            mask_token=None,
            oov_token="[OOV]",
            vocabulary=vocab_data,
            vocabulary_dtype=tf.string,
        )
        ds = tf.data.Dataset.from_tensor_slices([["earth"], ["wind"], ["and"]])
        ds = ds.map(layer)
        self.assertAllEqual(list(ds.as_numpy_iterator()), [[0], [1], [2]])

    def test_dataset_map_output_layer_created_in_function(self):
        vocab_data = ["earth", "wind", "and", "fire"]

        def apply_lookup(data):
            layer = index_lookup.IndexLookup(
                max_tokens=None,
                num_oov_indices=0,
                mask_token=None,
                oov_token="[OOV]",
                vocabulary=vocab_data,
                vocabulary_dtype=tf.string,
            )
            return layer(data)

        ds = tf.data.Dataset.from_tensor_slices([["earth"], ["wind"], ["and"]])
        ds = ds.map(apply_lookup)
        self.assertAllEqual(list(ds.as_numpy_iterator()), [[0], [1], [2]])


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupVocabularyTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_int_output_explicit_vocab(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_int_output_explicit_vocab_with_special_tokens(self):
        vocab_data = ["", "[OOV]", "earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_get_vocabulary_no_special_tokens(self):
        vocab_data = ["", "[OOV]", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary(include_special_tokens=False)
        self.assertAllEqual(returned_vocab, ["wind", "and", "fire"])
        self.assertAllEqual(layer.vocabulary_size(), 5)

    def test_vocab_multi_oov(self):
        vocab_data = ["", "[OOV]", "[OOV]", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=2,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(returned_vocab, vocab_data)

    def test_vocab_multi_oov_not_present(self):
        vocab_data = ["wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=10,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(
            returned_vocab, [""] + ["[OOV]"] * 10 + ["wind", "and", "fire"]
        )

    def test_vocab_with_max_cap(self):
        vocab_data = ["", "[OOV]", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)
        self.assertAllEqual(layer.vocabulary_size(), 5)

    def test_int_vocab_with_max_cap(self):
        vocab_data = [0, -1, 42, 1276, 1138]
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)
        self.assertAllEqual(layer.vocabulary_size(), 5)

    def test_vocab_with_multiple_oov_indices(self):
        vocab_data = ["", "[OOV]", "[OOV]", "[OOV]", "wind"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=3,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)

    def test_int_vocab_with_multiple_oov_indices(self):
        vocab_data = [0, -1, -1, -1, 42]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=3,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)

    def test_non_unique_vocab_fails(self):
        vocab_data = ["earth", "wind", "and", "fire", "fire"]
        with self.assertRaisesRegex(ValueError, "repeated term.*fire"):
            _ = index_lookup.IndexLookup(
                vocabulary=vocab_data,
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                vocabulary_dtype=tf.string,
            )

    def test_vocab_with_repeated_element_fails(self):
        vocab_data = ["earth", "earth", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        with self.assertRaisesRegex(ValueError, "repeated term.*earth"):
            layer.set_vocabulary(vocab_data)

    def test_vocab_with_reserved_oov_element_and_invert_true_fails(self):
        vocab_data = ["earth", "test", "[OOV]", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            invert=True,
            vocabulary_dtype=tf.string,
        )
        with self.assertRaisesRegex(ValueError, "reserved OOV"):
            layer.set_vocabulary(vocab_data)

    def test_vocab_with_reserved_mask_element_fails(self):
        vocab_data = ["earth", "mask_token", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="mask_token",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        with self.assertRaisesRegex(ValueError, "reserved mask"):
            layer.set_vocabulary(vocab_data)

    def test_vocab_size_changed_pad_to_max_false_fails(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            pad_to_max_tokens=False,
            output_mode=index_lookup.MULTI_HOT,
            vocabulary_dtype=tf.string,
        )
        layer.set_vocabulary(vocab_data)
        # Calling the layer should lock the vocabulary size.
        _ = layer([["earth"]])
        with self.assertRaisesRegex(
            RuntimeError, "vocabulary size cannot be changed"
        ):
            layer.set_vocabulary(vocab_data[:2])

    def test_vocab_with_idf_weights_non_tfidf_output_fails(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        weight_data = [1, 1, 1, 1, 1]
        with self.assertRaisesRegex(
            ValueError, "`idf_weights` should only be set if"
        ):
            index_lookup.IndexLookup(
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                output_mode=index_lookup.MULTI_HOT,
                vocabulary_dtype=tf.string,
                vocabulary=vocab_data,
                idf_weights=weight_data,
            )

    def test_vocab_with_idf_weights_length_mismatch_fails(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        weight_data = [1, 1, 1, 1, 1]  # too long
        with self.assertRaisesRegex(
            ValueError, "`idf_weights` must be the same length as vocab"
        ):
            index_lookup.IndexLookup(
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                output_mode=index_lookup.TF_IDF,
                vocabulary_dtype=tf.string,
                vocabulary=vocab_data,
                idf_weights=weight_data,
            )

    def test_vocab_without_idf_weights_tfidf_output_fails(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        with self.assertRaisesRegex(
            ValueError, "`idf_weights` must be set if output_mode is TF_IDF"
        ):
            index_lookup.IndexLookup(
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                output_mode=index_lookup.TF_IDF,
                vocabulary_dtype=tf.string,
                vocabulary=vocab_data,
            )

    def test_non_unique_int_vocab_fails(self):
        vocab_data = [12, 13, 14, 15, 15]
        with self.assertRaisesRegex(ValueError, "repeated term.*15"):
            _ = index_lookup.IndexLookup(
                vocabulary=vocab_data,
                max_tokens=None,
                num_oov_indices=1,
                mask_token=0,
                oov_token=-1,
                vocabulary_dtype=tf.int64,
            )

    def test_int_vocab_with_reserved_oov_element_and_invert_true_fails(self):
        vocab_data = [14, 38, -1, 34, 3, 84]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            invert=True,
            vocabulary_dtype=tf.int64,
        )
        with self.assertRaisesRegex(ValueError, "reserved OOV"):
            layer.set_vocabulary(vocab_data)

    def test_int_vocab_with_reserved_mask_element_fails(self):
        vocab_data = [125, 0, 3, 4, 94]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
        )
        with self.assertRaisesRegex(ValueError, "reserved mask"):
            layer.set_vocabulary(vocab_data)

    def test_no_vocab_file_string_fails(self):
        with self.assertRaisesRegex(ValueError, "non_existent_file"):
            _ = index_lookup.IndexLookup(
                vocabulary="non_existent_file",
                max_tokens=None,
                num_oov_indices=1,
                mask_token=0,
                oov_token=-1,
                vocabulary_dtype=tf.int64,
            )


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupInverseVocabularyTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_int_output_explicit_vocab(self):
        vocab_data = ["", "[OOV]", "earth", "wind", "and", "fire"]
        input_array = np.array([[2, 3, 4, 5], [5, 4, 2, 1]])
        expected_output = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "[OOV]"],
            ]
        )

        input_data = keras.Input(shape=(None,), dtype=tf.int64)
        layer = index_lookup.IndexLookup(
            vocabulary=vocab_data,
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            invert=True,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(expected_output, output_dataset)

    def test_vocab_with_max_cap(self):
        vocab_data = ["", "[OOV]", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            invert=True,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)

    def test_int_vocab_with_max_cap(self):
        vocab_data = [0, -1, 42, 1276, 1138]
        layer = index_lookup.IndexLookup(
            max_tokens=5,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
            invert=True,
        )
        layer.set_vocabulary(vocab_data)
        returned_vocab = layer.get_vocabulary()
        self.assertAllEqual(vocab_data, returned_vocab)

    def test_non_unique_vocab_fails(self):
        vocab_data = ["earth", "wind", "and", "fire", "fire"]
        with self.assertRaisesRegex(ValueError, "repeated term.*fire"):
            _ = index_lookup.IndexLookup(
                vocabulary=vocab_data,
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                vocabulary_dtype=tf.string,
                invert=True,
            )

    def test_non_int_output_fails(self):
        with self.assertRaisesRegex(
            ValueError, "`output_mode` must be `'int'`"
        ):
            _ = index_lookup.IndexLookup(
                max_tokens=None,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                vocabulary_dtype=tf.string,
                output_mode=index_lookup.COUNT,
                invert=True,
            )

    def test_vocab_with_repeated_element_fails(self):
        vocab_data = ["earth", "earth", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            invert=True,
        )
        with self.assertRaisesRegex(ValueError, "repeated term.*earth"):
            layer.set_vocabulary(vocab_data)

    def test_vocab_with_reserved_mask_element_fails(self):
        vocab_data = ["earth", "mask_token", "wind", "and", "fire"]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="mask_token",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            invert=True,
        )
        with self.assertRaisesRegex(ValueError, "reserved mask"):
            layer.set_vocabulary(vocab_data)

    def test_non_unique_int_vocab_fails(self):
        vocab_data = [12, 13, 14, 15, 15]
        with self.assertRaisesRegex(ValueError, "repeated term.*15"):
            _ = index_lookup.IndexLookup(
                vocabulary=vocab_data,
                max_tokens=None,
                num_oov_indices=1,
                mask_token=0,
                oov_token=-1,
                vocabulary_dtype=tf.int64,
                invert=True,
            )

    def test_int_vocab_with_repeated_element_fails(self):
        vocab_data = [11, 11, 34, 23, 124]
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=0,
            oov_token=-1,
            vocabulary_dtype=tf.int64,
            invert=True,
        )
        with self.assertRaisesRegex(ValueError, "repeated term.*11"):
            layer.set_vocabulary(vocab_data)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupErrorTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_too_long_vocab_fails_in_single_setting(self):
        vocab_data = ["earth", "wind", "and", "fire"]

        layer = index_lookup.IndexLookup(
            max_tokens=4,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
        )
        with self.assertRaisesRegex(
            ValueError, "vocabulary larger than the maximum vocab"
        ):
            layer.set_vocabulary(vocab_data)

    def test_zero_max_tokens_fails(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            _ = index_lookup.IndexLookup(
                max_tokens=0,
                num_oov_indices=1,
                mask_token="",
                oov_token="[OOV]",
                vocabulary_dtype=tf.string,
            )


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IndexLookupSavingTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def _write_to_temp_file(self, file_name, vocab_list):
        vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
        with tf.io.gfile.GFile(vocab_path, "w") as writer:
            for vocab in vocab_list:
                writer.write(vocab + "\n")
            writer.flush()
            writer.close()
        return vocab_path

    def test_vocabulary_persistence_across_saving(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]
        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        with self.subTest("keras_v3"):
            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_model.keras"
            )
            model.save(output_path, save_format="keras_v3")

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

        with self.subTest("savedmodel"):
            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_saved_model"
            )
            model.save(output_path, save_format="tf")

            # Delete the session and graph to ensure that the loaded model is
            # generated from scratch.
            keras.backend.clear_session()
            tf.io.gfile.remove(vocab_file)

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

    def test_vocabulary_persistence_file_across_cloning(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]
        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        # Clone the model and set weights.
        new_model = keras.models.clone_model(model)
        new_model.set_weights(model.get_weights())

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, new_model)

        # Validate correctness of the new model.
        new_output_dataset = new_model.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_persistence_file_vocabs_tf_save_tf_load(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
        tf.saved_model.save(obj=model, export_dir=output_path)

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()

        loaded_model = tf.saved_model.load(output_path)
        f = loaded_model.signatures["serving_default"]

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = f(tf.constant(input_array))["index_lookup"]
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_vocabulary_persistence_file_vocab_keras_save_tf_load(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
        model.save(output_path, save_format="tf")

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()

        loaded_model = tf.saved_model.load(output_path)
        f = loaded_model.signatures["serving_default"]

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = f(tf.constant(input_array))["index_lookup"]
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_persistence_file_vocab_keras_save_keras_load(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        with self.subTest("keras_v3"):
            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_model.keras"
            )
            model.save(output_path, save_format="keras_v3")

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

            # Try re-saving the layer. This simulates saving a layer
            # contained at a hub Module.
            input_data_2 = keras.Input(shape=(None,), dtype=tf.string)
            output_2 = loaded_model(input_data_2)
            model_2 = keras.Model(inputs=input_data_2, outputs=output_2)
            new_output_dataset = model_2.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_model_2.keras"
            )
            model_2.save(output_path, save_format="keras_v3")

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

        with self.subTest("saved_model"):
            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_saved_model"
            )
            model.save(output_path, save_format="tf")

            # Delete the session and graph to ensure that the loaded model is
            # generated from scratch.
            keras.backend.clear_session()
            tf.io.gfile.remove(vocab_file)

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

            # Try re-saving the layer. This simulates saving a layer
            # contained at a hub Module.
            input_data_2 = keras.Input(shape=(None,), dtype=tf.string)
            output_2 = loaded_model(input_data_2)
            model_2 = keras.Model(inputs=input_data_2, outputs=output_2)
            new_output_dataset = model_2.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

            # Save the model to disk.
            output_path = os.path.join(
                self.get_temp_dir(), "tf_keras_saved_model_2"
            )
            model_2.save(output_path, save_format="tf")

            # Delete the session and graph to ensure that the loaded model is
            # generated from scratch.
            keras.backend.clear_session()

            loaded_model = keras.models.load_model(
                output_path,
                custom_objects={"IndexLookup": index_lookup.IndexLookup},
            )

            # Ensure that the loaded model is unique
            # (so that the save/load is real)
            self.assertIsNot(model, loaded_model)

            # Validate correctness of the new model.
            new_output_dataset = loaded_model.predict(input_array)
            self.assertAllEqual(new_output_dataset, expected_output)

    def test_persistence_file_vocab_keras_save_keras_load_tf_save_tf_load(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
        model.save(output_path, save_format="tf")

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()
        tf.io.gfile.remove(vocab_file)

        loaded_model = keras.models.load_model(
            output_path,
            custom_objects={"IndexLookup": index_lookup.IndexLookup},
        )

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = loaded_model.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

        # Try re-saving the layer. This simulates saving a layer contained at
        # a hub Module.
        input_data_2 = keras.Input(shape=(None,), dtype=tf.string)
        output_2 = loaded_model(input_data_2)
        model_2 = keras.Model(inputs=input_data_2, outputs=output_2)
        new_output_dataset = model_2.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(
            self.get_temp_dir(), "tf_keras_saved_model_2"
        )
        tf.saved_model.save(model_2, output_path)

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()

        loaded_model = tf.saved_model.load(output_path)
        f = loaded_model.signatures["serving_default"]

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = f(tf.constant(input_array))["model"]
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_persistence_file_vocab_keras_save_keras_load_keras_save_keras_load(
        self,
    ):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )
        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        vocab_file = self._write_to_temp_file("temp", vocab_data)

        # Build and validate a golden model.
        input_data = keras.Input(shape=(None,), dtype=tf.string)
        layer = index_lookup.IndexLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        int_data = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_array)
        self.assertAllEqual(output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
        model.save(output_path, save_format="tf")

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()
        tf.io.gfile.remove(vocab_file)

        loaded_model = keras.models.load_model(
            output_path,
            custom_objects={"IndexLookup": index_lookup.IndexLookup},
        )

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = loaded_model.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

        # Try re-saving the layer. This simulates saving a layer contained at
        # a hub Module.
        input_data_2 = keras.Input(shape=(None,), dtype=tf.string)
        output_2 = loaded_model(input_data_2)
        model_2 = keras.Model(inputs=input_data_2, outputs=output_2)
        new_output_dataset = model_2.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

        # Save the model to disk.
        output_path = os.path.join(
            self.get_temp_dir(), "tf_keras_saved_model_2"
        )
        model_2.save(output_path, save_format="tf")

        # Delete the session and graph to ensure that the loaded model is
        # generated from scratch.
        keras.backend.clear_session()

        loaded_model = keras.models.load_model(
            output_path,
            custom_objects={"IndexLookup": index_lookup.IndexLookup},
        )

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_dataset = model_2.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_static_table_config_weight_data_transfer_succeeds(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        vocab_file = self._write_to_temp_file("temp", vocab_data)
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )

        expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

        # Build and validate a golden model.
        layer_cls = index_lookup.IndexLookup
        layer = layer_cls(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_file,
        )
        config = layer.get_config()
        weights = layer.get_weights()

        layer = layer_cls.from_config(config)
        layer.set_weights(weights)

        input_data = keras.Input(shape=(None,), dtype=tf.string)
        output = layer(input_data)
        model = keras.Model(inputs=input_data, outputs=output)

        new_output_dataset = model.predict(input_array)
        self.assertAllEqual(new_output_dataset, expected_output)

    def test_sparse_output_across_saving(self):
        vocab_data = ["earth", "wind", "and", "fire"]
        input_array = np.array(
            [
                ["earth", "wind", "and", "fire"],
                ["fire", "and", "earth", "michigan"],
            ]
        )

        expected_output = [[0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0]]

        layer_cls = index_lookup.IndexLookup
        layer = layer_cls(
            max_tokens=None,
            num_oov_indices=1,
            mask_token="",
            oov_token="[OOV]",
            vocabulary_dtype=tf.string,
            vocabulary=vocab_data,
            output_mode="multi_hot",
            sparse=True,
        )
        config = layer.get_config()
        layer = layer_cls.from_config(config)

        output = layer(input_array)
        self.assertIsInstance(output, tf.SparseTensor)
        self.assertAllEqual(tf.sparse.to_dense(output), expected_output)


class EagerExecutionDisabled(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_lookup(self):
        # We need this test for model_to_estimator followed by
        # export_saved_model, which will call the layer in a legacy session.
        # This could also happen directly if a user calls disable_v2_behavior or
        # disable_eager_execution.
        with tf.compat.v1.Session():
            with test_utils.run_eagerly_scope(False):
                vocab_data = ["earth", "wind", "and", "fire"]
                input_array = np.array(["earth", "wind", "and", "fire"])
                expected_output = [1, 2, 3, 4]

                input_data = keras.Input(shape=(None,), dtype=tf.string)
                layer = index_lookup.IndexLookup(
                    max_tokens=None,
                    num_oov_indices=1,
                    mask_token=None,
                    oov_token="[OOV]",
                    vocabulary_dtype=tf.string,
                    vocabulary=vocab_data,
                )
                int_data = layer(input_data)
                model = keras.Model(inputs=input_data, outputs=int_data)
                # In a TF1 session the user will need to make sure all tables
                # are initialized themselves.
                tf.compat.v1.tables_initializer().run()
                output_dataset = model(input_array)
                self.assertAllEqual(output_dataset, expected_output)


if __name__ == "__main__":
    # IndexLookup is only exported as a TF2 API.
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
