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
"""Tests for RNN cell wrappers."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import layers
from keras.layers.rnn import cell_wrappers
from keras.layers.rnn import legacy_cells
from keras.legacy_tf_layers import base as legacy_base_layer
from keras.testing_infra import test_combinations
from keras.utils import generic_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class RNNCellWrapperTest(tf.test.TestCase, parameterized.TestCase):
    def testResidualWrapper(self):
        wrapper_type = cell_wrappers.ResidualWrapper
        x = tf.convert_to_tensor(np.array([[1.0, 1.0, 1.0]]), dtype="float32")
        m = tf.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]), dtype="float32")
        base_cell = legacy_cells.GRUCell(
            3,
            kernel_initializer=tf.compat.v1.constant_initializer(0.5),
            bias_initializer=tf.compat.v1.constant_initializer(0.5),
        )
        g, m_new = base_cell(x, m)
        wrapper_object = wrapper_type(base_cell)
        self.assertDictEqual(
            {"cell": base_cell}, wrapper_object._trackable_children()
        )
        wrapper_object.get_config()  # Should not throw an error

        g_res, m_new_res = wrapper_object(x, m)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
        res = self.evaluate([g, g_res, m_new, m_new_res])
        # Residual connections
        self.assertAllClose(res[1], res[0] + [1.0, 1.0, 1.0])
        # States are left untouched
        self.assertAllClose(res[2], res[3])

    def testResidualWrapperWithSlice(self):
        wrapper_type = cell_wrappers.ResidualWrapper
        x = tf.convert_to_tensor(
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), dtype="float32"
        )
        m = tf.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]), dtype="float32")
        base_cell = legacy_cells.GRUCell(
            3,
            kernel_initializer=tf.compat.v1.constant_initializer(0.5),
            bias_initializer=tf.compat.v1.constant_initializer(0.5),
        )
        g, m_new = base_cell(x, m)

        def residual_with_slice_fn(inp, out):
            inp_sliced = tf.slice(inp, [0, 0], [-1, 3])
            return inp_sliced + out

        g_res, m_new_res = wrapper_type(base_cell, residual_with_slice_fn)(x, m)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
        res_g, res_g_res, res_m_new, res_m_new_res = self.evaluate(
            [g, g_res, m_new, m_new_res]
        )
        # Residual connections
        self.assertAllClose(res_g_res, res_g + [1.0, 1.0, 1.0])
        # States are left untouched
        self.assertAllClose(res_m_new, res_m_new_res)

    def testDeviceWrapper(self):
        wrapper_type = cell_wrappers.DeviceWrapper
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 3])
        cell = legacy_cells.GRUCell(3)
        wrapped_cell = wrapper_type(cell, "/cpu:0")
        self.assertDictEqual({"cell": cell}, wrapped_cell._trackable_children())
        wrapped_cell.get_config()  # Should not throw an error

        outputs, _ = wrapped_cell(x, m)
        self.assertIn("cpu:0", outputs.device.lower())

    @parameterized.parameters(
        [cell_wrappers.DropoutWrapper, cell_wrappers.ResidualWrapper]
    )
    def testWrapperKerasStyle(self, wrapper):
        """Tests if wrapper cell is instantiated in keras style scope."""
        wrapped_cell = wrapper(legacy_cells.BasicRNNCell(1))
        self.assertIsNone(getattr(wrapped_cell, "_keras_style", None))

    @parameterized.parameters(
        [cell_wrappers.DropoutWrapper, cell_wrappers.ResidualWrapper]
    )
    def testWrapperWeights(self, wrapper):
        """Tests that wrapper weights contain wrapped cells weights."""
        base_cell = layers.SimpleRNNCell(1, name="basic_rnn_cell")
        rnn_cell = wrapper(base_cell)
        rnn_layer = layers.RNN(rnn_cell)
        inputs = tf.convert_to_tensor([[[1]]], dtype=tf.float32)
        rnn_layer(inputs)

        wrapper_name = generic_utils.to_snake_case(wrapper.__name__)
        expected_weights = [
            "rnn/" + wrapper_name + "/" + var
            for var in ("kernel:0", "recurrent_kernel:0", "bias:0")
        ]
        self.assertLen(rnn_cell.weights, 3)
        self.assertCountEqual(
            [v.name for v in rnn_cell.weights], expected_weights
        )
        self.assertCountEqual(
            [v.name for v in rnn_cell.trainable_variables], expected_weights
        )
        self.assertCountEqual(
            [v.name for v in rnn_cell.non_trainable_variables], []
        )
        self.assertCountEqual(
            [v.name for v in rnn_cell.cell.weights], expected_weights
        )

    @parameterized.parameters(
        [cell_wrappers.DropoutWrapper, cell_wrappers.ResidualWrapper]
    )
    def testWrapperV2Caller(self, wrapper):
        """Tests that wrapper V2 is using the LayerRNNCell's caller."""

        with legacy_base_layer.keras_style_scope():
            base_cell = legacy_cells.MultiRNNCell(
                [legacy_cells.BasicRNNCell(1) for _ in range(2)]
            )
        rnn_cell = wrapper(base_cell)
        inputs = tf.convert_to_tensor([[1]], dtype=tf.float32)
        state = tf.convert_to_tensor([[1]], dtype=tf.float32)
        _ = rnn_cell(inputs, [state, state])
        weights = base_cell._cells[0].weights
        self.assertLen(weights, expected_len=2)
        self.assertTrue(all("_wrapper" in v.name for v in weights))

    @parameterized.parameters(
        [cell_wrappers.DropoutWrapper, cell_wrappers.ResidualWrapper]
    )
    def testWrapperV2Build(self, wrapper):
        cell = legacy_cells.LSTMCell(10)
        wrapper = wrapper(cell)
        wrapper.build((1,))
        self.assertTrue(cell.built)

    def testDeviceWrapperSerialization(self):
        wrapper_cls = cell_wrappers.DeviceWrapper
        cell = layers.LSTMCell(10)
        wrapper = wrapper_cls(cell, "/cpu:0")
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        self.assertDictEqual(config, reconstructed_wrapper.get_config())
        self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    def testResidualWrapperSerialization(self):
        wrapper_cls = cell_wrappers.ResidualWrapper
        cell = layers.LSTMCell(10)
        wrapper = wrapper_cls(cell)
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        self.assertDictEqual(config, reconstructed_wrapper.get_config())
        self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

        wrapper = wrapper_cls(cell, residual_fn=lambda i, o: i + i + o)
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        # Assert the reconstructed function will perform the math correctly.
        self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 4)

        def residual_fn(inputs, outputs):
            return inputs * 3 + outputs

        wrapper = wrapper_cls(cell, residual_fn=residual_fn)
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        # Assert the reconstructed function will perform the math correctly.
        self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 5)

    def testDropoutWrapperSerialization(self):
        wrapper_cls = cell_wrappers.DropoutWrapper
        cell = layers.GRUCell(10)
        wrapper = wrapper_cls(cell)
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        self.assertDictEqual(config, reconstructed_wrapper.get_config())
        self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

        wrapper = wrapper_cls(cell, dropout_state_filter_visitor=lambda s: True)
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        self.assertTrue(reconstructed_wrapper._dropout_state_filter(None))

        def dropout_state_filter_visitor(unused_state):
            return False

        wrapper = wrapper_cls(
            cell, dropout_state_filter_visitor=dropout_state_filter_visitor
        )
        config = wrapper.get_config()

        reconstructed_wrapper = wrapper_cls.from_config(config)
        self.assertFalse(reconstructed_wrapper._dropout_state_filter(None))

    def testDropoutWrapperWithKerasLSTMCell(self):
        wrapper_cls = cell_wrappers.DropoutWrapper
        cell = layers.LSTMCell(10)

        with self.assertRaisesRegex(ValueError, "does not work with "):
            wrapper_cls(cell)

        cell = layers.LSTMCellV2(10)
        with self.assertRaisesRegex(ValueError, "does not work with "):
            wrapper_cls(cell)


if __name__ == "__main__":
    tf.test.main()
