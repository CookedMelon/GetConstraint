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

"""Tests the JSON encoder and decoder."""

import enum

import tensorflow.compat.v2 as tf

from keras.saving.legacy.saved_model import json_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


class JsonUtilsTest(test_combinations.TestCase):
    def test_encode_decode_tensor_shape(self):
        metadata = {
            "key1": tf.TensorShape(None),
            "key2": [tf.TensorShape([None]), tf.TensorShape([3, None, 5])],
        }
        string = json_utils.Encoder().encode(metadata)
        loaded = json_utils.decode(string)

        self.assertEqual(set(loaded.keys()), {"key1", "key2"})
        self.assertAllEqual(loaded["key1"].rank, None)
        self.assertAllEqual(loaded["key2"][0].as_list(), [None])
        self.assertAllEqual(loaded["key2"][1].as_list(), [3, None, 5])

    def test_encode_decode_tuple(self):
        metadata = {"key1": (3, 5), "key2": [(1, (3, 4)), (1,)]}
        string = json_utils.Encoder().encode(metadata)
        loaded = json_utils.decode(string)

        self.assertEqual(set(loaded.keys()), {"key1", "key2"})
        self.assertAllEqual(loaded["key1"], (3, 5))
        self.assertAllEqual(loaded["key2"], [(1, (3, 4)), (1,)])

    def test_encode_decode_type_spec(self):
        spec = tf.TensorSpec((1, 5), tf.float32)
        string = json_utils.Encoder().encode(spec)
        loaded = json_utils.decode(string)
        self.assertEqual(spec, loaded)

        invalid_type_spec = {
            "class_name": "TypeSpec",
            "type_spec": "Invalid Type",
            "serialized": None,
        }
        string = json_utils.Encoder().encode(invalid_type_spec)
        with self.assertRaisesRegexp(
            ValueError, "No TypeSpec has been registered"
        ):
            loaded = json_utils.decode(string)

    def test_encode_decode_enum(self):
        class Enum(enum.Enum):
            CLASS_A = "a"
            CLASS_B = "b"

        config = {"key": Enum.CLASS_A, "key2": Enum.CLASS_B}
        string = json_utils.Encoder().encode(config)
        loaded = json_utils.decode(string)
        self.assertAllEqual({"key": "a", "key2": "b"}, loaded)

    @test_utils.run_v2_only
    def test_encode_decode_ragged_tensor(self):
        x = tf.ragged.constant([[1.0, 2.0], [3.0]])
        string = json_utils.Encoder().encode(x)
        loaded = json_utils.decode(string)
        self.assertAllEqual(loaded, x)

    @test_utils.run_v2_only
    def test_encode_decode_extension_type_tensor(self):
        class MaskedTensor(tf.experimental.ExtensionType):
            __name__ = "MaskedTensor"
            values: tf.Tensor
            mask: tf.Tensor

        x = MaskedTensor(
            values=[[1, 2, 3], [4, 5, 6]],
            mask=[[True, True, False], [True, False, True]],
        )
        string = json_utils.Encoder().encode(x)
        loaded = json_utils.decode(string)
        self.assertAllEqual(loaded, x)

    def test_encode_decode_bytes(self):
        b_string = b"abc"
        json_string = json_utils.Encoder().encode(b_string)
        loaded = json_utils.decode(json_string)
        self.assertAllEqual(b_string, loaded)


if __name__ == "__main__":
    tf.test.main()
