# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for trackable object SavedModel save."""

import os

import tensorflow.compat.v2 as tf

from keras.layers import core
from keras.optimizers.legacy import adam

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


class _ModelWithOptimizerUsingDefun(tf.train.Checkpoint):
    def __init__(self):
        self.dense = core.Dense(1)
        self.optimizer = adam.Adam(0.01)

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, 2], tf.float32),
            tf.TensorSpec([None], tf.float32),
        ),
    )
    def call(self, x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self.dense(x) - y) ** 2.0)
        trainable_variables = self.dense.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {"loss": loss}


class MemoryTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._model = _ModelWithOptimizerUsingDefun()

    @tf_test_utils.assert_no_garbage_created
    def DISABLED_test_no_reference_cycles(self):
        x = tf.constant([[3.0, 4.0]])
        y = tf.constant([2.0])
        self._model.call(x, y)
        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        tf.saved_model.save(self._model, save_dir, self._model.call)


if __name__ == "__main__":
    tf.test.main()
