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
"""Tests for compile utitilies."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras import losses as losses_mod
from keras import metrics as metrics_mod
from keras.engine import compile_utils
from keras.testing_infra import test_combinations


class LossesContainerTest(test_combinations.TestCase):
    def test_single_loss(self):
        loss_container = compile_utils.LossesContainer("mse")
        y_t, y_p = tf.ones((10, 5)), tf.zeros((10, 5))
        total_loss = loss_container(y_t, y_p)

        self.assertTrue(loss_container._built)
        self.assertLen(loss_container._losses, 1)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.numpy(), 1.0)
        self.assertLen(loss_container.metrics, 1)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 1.0)

        loss_container.reset_state()
        self.assertEqual(loss_metric.result().numpy(), 0.0)

    def test_loss_list(self):
        loss_container = compile_utils.LossesContainer(["mse", "mae"], [1, 0.5])

        y_t = [tf.ones((10, 1)), tf.zeros((10, 1))]
        y_p = [tf.ones((10, 1)), tf.ones((10, 1))]
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)

        self.assertEqual(loss_container._output_names, ["output_1", "output_2"])

        self.assertLen(loss_container._losses, 2)
        self.assertEqual(total_loss.numpy(), 0.25)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.25)

        output_1_metric = loss_container.metrics[1]
        self.assertEqual(output_1_metric.name, "output_1_loss")
        self.assertEqual(output_1_metric.result().numpy(), 0)

        output_2_metric = loss_container.metrics[2]
        self.assertEqual(output_2_metric.name, "output_2_loss")
        self.assertEqual(output_2_metric.result().numpy(), 0.5)

        loss_container.reset_state()
        self.assertEqual(loss_metric.result().numpy(), 0)
        self.assertEqual(output_1_metric.result().numpy(), 0)
        self.assertEqual(output_2_metric.result().numpy(), 0)

    def test_loss_dict(self):
        loss_container = compile_utils.LossesContainer(
            {"out1": "mse", "out2": "mae"}, {"out1": 1, "out2": 0.5}
        )

        y_t = {"out1": tf.ones((10, 1)), "out2": tf.zeros((10, 1))}
        y_p = {"out1": tf.ones((10, 1)), "out2": tf.ones((10, 1))}
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)

        self.assertLen(loss_container._losses, 2)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.numpy(), 0.25)
        self.assertLen(loss_container.metrics, 3)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.25)

        out1_metric = loss_container.metrics[1]
        self.assertEqual(out1_metric.name, "out1_loss")
        self.assertEqual(out1_metric.result().numpy(), 0)

        out2_metric = loss_container.metrics[2]
        self.assertEqual(out2_metric.name, "out2_loss")
        self.assertEqual(out2_metric.result().numpy(), 0.5)

        loss_container.reset_state()
        self.assertEqual(loss_metric.result().numpy(), 0)
        self.assertEqual(out1_metric.result().numpy(), 0)
        self.assertEqual(out2_metric.result().numpy(), 0)

    def test_loss_partial_dict_with_output_names(self):
        loss_container = compile_utils.LossesContainer(
            {"out2": "mae"}, {"out2": 1.0}, output_names=["out1", "out2"]
        )

        y_t = [tf.ones((10, 1)), tf.zeros((10, 1))]
        y_p = [tf.ones((10, 1)), tf.ones((10, 1))]
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)

        self.assertEqual(total_loss.numpy(), 0.5)
        self.assertLen(loss_container.metrics, 2)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.5)

        out2_metric = loss_container.metrics[1]
        self.assertEqual(out2_metric.name, "out2_loss")
        self.assertEqual(out2_metric.result().numpy(), 0.5)

    def test_loss_dict_with_nones(self):
        loss_container = compile_utils.LossesContainer(
            {"out1": None, "out2": "mae"}
        )

        y_t = {"out1": tf.ones((10, 1)), "out2": tf.zeros((10, 1))}
        y_p = {"out1": tf.ones((10, 1)), "out2": tf.ones((10, 1))}
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)

        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.numpy(), 0.5)
        self.assertLen(loss_container.metrics, 2)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.5)

        out2_metric = loss_container.metrics[1]
        self.assertEqual(out2_metric.name, "out2_loss")
        self.assertEqual(out2_metric.result().numpy(), 0.5)

    def test_nested_structure(self):
        loss_container = compile_utils.LossesContainer(
            {"b": ["mse", None], "a": "mae"},
            loss_weights={"b": [0.5, 0], "a": 1},
        )

        y_t = {
            "b": [tf.ones((10, 1)), tf.zeros((10, 1))],
            "a": tf.zeros((10, 1)),
        }
        y_p = {
            "b": [tf.zeros((10, 1)), tf.zeros((10, 1))],
            "a": tf.ones((10, 1)),
        }
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.numpy(), 0.75)
        self.assertLen(loss_container.metrics, 3)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.75)

        a_metric = loss_container.metrics[1]
        self.assertEqual(a_metric.name, "a_loss")
        self.assertEqual(a_metric.result().numpy(), 0.5)

        b_1_metric = loss_container.metrics[2]
        self.assertEqual(b_1_metric.name, "b_1_loss")
        self.assertEqual(b_1_metric.result().numpy(), 0.5)

    def test_no_input_mutation(self):
        loss = {"a": "mae"}
        loss_container = compile_utils.LossesContainer(loss)

        y_t = {"a": tf.zeros((10, 1))}
        y_p = {"a": tf.ones((10, 1)), "b": tf.zeros((10, 1))}
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.numpy(), 0.5)
        self.assertLen(loss, 1)

    def test_broadcast_single_loss(self):
        loss_container = compile_utils.LossesContainer("mse")

        y_t = [tf.ones((10, 1)), tf.zeros((10, 1))]
        y_p = [tf.ones((10, 1)), tf.ones((10, 1))]
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        total_loss = loss_container(y_t, y_p, sample_weight=sw)
        self.assertEqual(total_loss.numpy(), 0.5)
        self.assertLen(loss_container.metrics, 3)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 0.5)

        output_1_metric = loss_container.metrics[1]
        self.assertEqual(output_1_metric.name, "output_1_loss")
        self.assertEqual(output_1_metric.result().numpy(), 0.0)

        output_2_metric = loss_container.metrics[2]
        self.assertEqual(output_2_metric.name, "output_2_loss")
        self.assertEqual(output_2_metric.result().numpy(), 0.5)

    def test_missing_label_with_no_loss(self):
        # It's ok to exclude a label if that label has no
        # losses or metrics associated with it.
        loss_container = compile_utils.LossesContainer(
            {"output1": "mse", "output3": "mae"}
        )

        y_p = {
            "output1": tf.convert_to_tensor([[0], [1], [2]]),
            "output2": tf.convert_to_tensor([[3], [4], [5]]),
            "output3": tf.convert_to_tensor([[6], [7], [8]]),
        }
        y_t = {
            "output1": tf.convert_to_tensor([[1], [2], [3]]),
            "output3": tf.convert_to_tensor([[4], [5], [6]]),
        }

        total_loss = loss_container(y_t, y_p)
        self.assertEqual(total_loss.numpy(), 3.0)
        self.assertLen(loss_container.metrics, 3)

        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertEqual(loss_metric.result().numpy(), 3.0)

        output_1_metric = loss_container.metrics[1]
        self.assertEqual(output_1_metric.name, "output1_loss")
        self.assertEqual(output_1_metric.result().numpy(), 1.0)

        output_3_metric = loss_container.metrics[2]
        self.assertEqual(output_3_metric.name, "output3_loss")
        self.assertEqual(output_3_metric.result().numpy(), 2.0)

    def test_mismatched_dtypes(self):
        y_t = tf.constant([1, 9, 2, -5], shape=(2, 2))
        y_p = tf.constant([4, 8, 12, 8], shape=(2, 2), dtype=tf.float32)

        def my_mae(labels, preds):
            self.assertEqual(labels.dtype, tf.int32)
            self.assertEqual(preds.dtype, tf.float32)
            labels = tf.cast(labels, preds.dtype)
            return backend.mean(tf.abs(preds - labels), axis=-1)

        loss_container = compile_utils.LossesContainer(my_mae)
        total_loss = loss_container(y_t, y_p)
        self.assertEqual(total_loss.dtype, tf.float32)

    def test_integer_dtypes(self):
        y_t = tf.constant([1, 9, 2, -5], shape=(2, 2))
        y_p = tf.constant([4, 8, 12, 8], shape=(2, 2), dtype=tf.int64)

        def my_mae(labels, preds):
            self.assertEqual(labels.dtype, tf.int64)
            self.assertEqual(preds.dtype, tf.int64)
            return backend.mean(tf.abs(preds - labels), axis=-1)

        loss_container = compile_utils.LossesContainer(my_mae)
        total_loss = loss_container(y_t, y_p)
        self.assertEqual(total_loss.dtype, tf.int64)

    def test_float_dtypes(self):
        y_t = tf.constant([1, 9, 2, -5], shape=(2, 2), dtype=tf.float32)
        y_p = tf.constant([4, 8, 12, 8], shape=(2, 2), dtype=tf.float64)

        def my_mae(labels, preds):
            self.assertEqual(labels.dtype, tf.float64)
            self.assertEqual(preds.dtype, tf.float64)
            return backend.mean(tf.abs(preds - labels), axis=-1)

        loss_container = compile_utils.LossesContainer(my_mae)
        total_loss = loss_container(y_t, y_p)
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(total_loss.dtype, tf.float64)

    @test_combinations.generate(
        test_combinations.combine(
            input_type=["dense", "masked", "ragged"],
            reduction=["auto", "sum"],
            use_sample_weights=[True, False],
        ),
    )
    def test_loss_consistency(self, input_type, reduction, use_sample_weights):
        y_p = tf.ragged.constant(
            [[[1], [1], [1]], [[1], [1]]], dtype=tf.float32
        )
        y_t = tf.ragged.constant(
            [[[1], [0], [0]], [[1], [1]]], dtype=tf.float32
        )

        if input_type == "masked":
            mask = tf.ones_like(y_p).to_tensor()
            y_p = y_p.to_tensor()
            y_t = y_t.to_tensor()
            y_p._keras_mask = mask
        elif input_type == "dense":
            y_p = y_p.to_tensor()
            y_t = y_t.to_tensor()

        if input_type == "dense":
            count = 6
        else:
            count = 5

        if use_sample_weights:
            wrong = 4
            maybe_sample_weight = {
                "sample_weight": tf.constant([[2], [1]], dtype=tf.float32)
            }
        else:
            wrong = 2
            maybe_sample_weight = {}

        expected = wrong
        if reduction != "sum":
            expected /= count

        loss_obj = losses_mod.MeanAbsoluteError(reduction=reduction)

        result = loss_obj(y_t, y_p, **maybe_sample_weight)
        self.assertAlmostEqual(result.numpy(), expected)

        container = compile_utils.LossesContainer(loss_obj)
        container_result = container(y_t, y_p, **maybe_sample_weight)
        self.assertAlmostEqual(container_result.numpy(), expected)

    def test_loss_masking(self):
        loss_container = compile_utils.LossesContainer("mae")
        y_p = tf.constant([[[1], [1]], [[0], [0]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        # Reduction is "sum_over_batch_size" that's not the literal batch size,
        # but the number of elements being summed: The number of valid
        # emlements. So since the mask has two valid items, the number of
        # elements is 2.
        y_p._keras_mask = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)

        total_loss = loss_container(y_t, y_p)
        self.assertAlmostEqual(total_loss.numpy(), 0.5)  # sum over num valid

        self.assertLen(loss_container.metrics, 1)
        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertAlmostEqual(loss_metric.result().numpy(), 0.5)

    def test_loss_sample_weight(self):
        loss_container = compile_utils.LossesContainer("mae")
        y_p = tf.constant([[[1], [1]], [[0], [0]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        sw = tf.constant([[0.2, 0.3], [0.5, 0]], dtype=tf.float32)

        total_loss = loss_container(y_t, y_p, sample_weight=sw)
        # (0 * .2 + 0 * .3 + 1 * .5 + 1 * 0) / 4
        self.assertAlmostEqual(total_loss.numpy(), 0.125)

        self.assertLen(loss_container.metrics, 1)
        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertAlmostEqual(loss_metric.result().numpy(), 0.125)

    def test_loss_masking_sample_weight(self):
        loss_container = compile_utils.LossesContainer("mae")
        y_p = tf.constant([[[1], [1]], [[0], [0]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        sw = tf.constant([[0.2, 0.3], [0.5, 0]], dtype=tf.float32)
        y_p._keras_mask = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)

        total_loss = loss_container(y_t, y_p, sample_weight=sw)
        # (0 * .2 + 1 * .5) / 2
        self.assertAlmostEqual(total_loss.numpy(), 0.25)  # sum over num valid

        self.assertLen(loss_container.metrics, 1)
        loss_metric = loss_container.metrics[0]
        self.assertEqual(loss_metric.name, "loss")
        self.assertAlmostEqual(loss_metric.result().numpy(), 0.25)

    def test_custom_loss_callables(self):
        def custom_loss_fn(y_true, y_pred):
            return tf.reduce_sum(y_true - y_pred)

        class CustomLossClass:
            def __call__(self, y_true, y_pred):
                return tf.reduce_sum(y_true - y_pred)

        loss_container = compile_utils.LossesContainer(
            [custom_loss_fn, CustomLossClass()]
        )
        y_t, y_p = tf.ones((10, 5)), tf.zeros((10, 5))
        loss_container(y_t, y_p)

        self.assertEqual(loss_container._losses[0].name, "custom_loss_fn")
        self.assertEqual(loss_container._losses[1].name, "custom_loss_class")

    def test_ragged_tensor_output(self):
        """Ensure ragged tensors can be passed as targets and predictions."""

        def custom_loss_fn(y_true, y_pred):
            """MSE supports RaggedTensors directly."""
            return losses_mod.mse(y_true, y_pred)

        class CustomLossClass(losses_mod.Loss):
            """User defined loss func must implement RaggedTensor support."""

            def call(self, y_true, y_pred):
                losses = tf.ragged.map_flat_values(
                    tf.math.squared_difference, y_true, y_pred
                )
                return tf.reduce_mean(losses)

        loss_container = compile_utils.LossesContainer(
            [custom_loss_fn, CustomLossClass()]
        )

        v_t = tf.constant([[3.0, 4.0], [1.0, 2.0], [3.0, 5.0]])
        v_p = tf.constant([[3.1, 4.0], [1.0, 2.0], [3.0, 5.0]])

        y_t = tf.expand_dims(tf.RaggedTensor.from_row_splits(v_t, [0, 2, 3]), 0)
        y_p = tf.expand_dims(tf.RaggedTensor.from_row_splits(v_p, [0, 2, 3]), 0)
        total_loss = loss_container(y_t, y_p)

        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertEqual(loss_container._losses[0].name, "custom_loss_fn")


class MetricsContainerTest(test_combinations.TestCase):
    def test_single_metric(self):
        metric_container = compile_utils.MetricsContainer("mse")
        y_t, y_p = tf.ones((10, 5)), tf.zeros((10, 5))
        metric_container.update_state(y_t, y_p)

        self.assertLen(metric_container.metrics, 1)
        metric = metric_container.metrics[0]
        self.assertEqual(metric.name, "mse")
        self.assertEqual(metric.result().numpy(), 1.0)

        metric_container.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)

    def test_list_of_metrics_one_output(self):
        metric_container = compile_utils.MetricsContainer(["mse", "mae"])
        y_t, y_p = 2 * tf.ones((10, 5)), tf.zeros((10, 5))
        metric_container.update_state(y_t, y_p)
        self.assertLen(metric_container.metrics, 2)

        mse_metric = metric_container.metrics[0]
        self.assertEqual(mse_metric.name, "mse")
        self.assertEqual(mse_metric.result().numpy(), 4.0)

        mae_metric = metric_container.metrics[1]
        self.assertEqual(mae_metric.name, "mae")
        self.assertEqual(mae_metric.result().numpy(), 2.0)

        metric_container.reset_state()
        self.assertEqual(mse_metric.result().numpy(), 0.0)
        self.assertEqual(mae_metric.result().numpy(), 0.0)

    def test_list_of_metrics_list_of_outputs(self):
        metric_container = compile_utils.MetricsContainer(
            metrics=["mse", "mae"],  # Should broadcast to both outputs.
            weighted_metrics=["accuracy"],
        )  # Should broadcast to both outputs.

        y_t = [tf.ones((10, 1)), tf.zeros((10, 1))]
        y_p = [tf.ones((10, 1)), 2 * tf.ones((10, 1))]
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        metric_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metric_container.metrics, 6)

        mse_metric = metric_container.metrics[0]
        self.assertEqual(mse_metric.name, "output_1_mse")
        self.assertEqual(mse_metric.result().numpy(), 0.0)

        mse_metric = metric_container.metrics[1]
        self.assertEqual(mse_metric.name, "output_1_mae")
        self.assertEqual(mse_metric.result().numpy(), 0.0)

        acc_metric_1 = metric_container.metrics[2]
        self.assertEqual(acc_metric_1.name, "output_1_accuracy")
        self.assertEqual(acc_metric_1.result().numpy(), 1.0)
        self.assertEqual(acc_metric_1._fn, metrics_mod.binary_accuracy)

        mae_metric = metric_container.metrics[3]
        self.assertEqual(mae_metric.name, "output_2_mse")
        self.assertEqual(mae_metric.result().numpy(), 4.0)

        mae_metric = metric_container.metrics[4]
        self.assertEqual(mae_metric.name, "output_2_mae")
        self.assertEqual(mae_metric.result().numpy(), 2.0)

        acc_metric_2 = metric_container.metrics[5]
        self.assertEqual(acc_metric_2.name, "output_2_accuracy")
        self.assertEqual(acc_metric_2.result().numpy(), 0.0)
        self.assertEqual(acc_metric_2._fn, metrics_mod.binary_accuracy)

        weighted_metrics = metric_container.weighted_metrics
        self.assertLen(weighted_metrics, 2)
        self.assertEqual(weighted_metrics[0].name, "output_1_accuracy")
        self.assertEqual(weighted_metrics[1].name, "output_2_accuracy")

        unweighted_metrics = metric_container.unweighted_metrics
        self.assertLen(unweighted_metrics, 4)
        self.assertEqual(unweighted_metrics[0].name, "output_1_mse")
        self.assertEqual(unweighted_metrics[1].name, "output_1_mae")
        self.assertEqual(unweighted_metrics[2].name, "output_2_mse")
        self.assertEqual(unweighted_metrics[3].name, "output_2_mae")

    def test_metric_dict(self):
        metric_container = compile_utils.MetricsContainer(
            metrics={"out1": "mse", "out2": "mae"},
            weighted_metrics={"out1": "mse", "out2": "mae"},
        )

        y_t = {"out1": tf.ones((10, 1)), "out2": tf.zeros((10, 1))}
        y_p = {"out1": tf.ones((10, 1)), "out2": 2 * tf.ones((10, 1))}
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        metric_container.update_state(y_t, y_p, sample_weight=sw)

        mse_metric = metric_container.metrics[0]
        self.assertEqual(mse_metric.name, "out1_mse")
        self.assertEqual(mse_metric.result().numpy(), 0.0)

        weighted_mse_metric = metric_container.metrics[1]
        self.assertEqual(weighted_mse_metric.name, "out1_weighted_mse")
        self.assertEqual(weighted_mse_metric.result().numpy(), 0.0)

        mae_metric = metric_container.metrics[2]
        self.assertEqual(mae_metric.name, "out2_mae")
        self.assertEqual(mae_metric.result().numpy(), 2.0)

        weighted_mae_metric = metric_container.metrics[3]
        self.assertEqual(weighted_mae_metric.name, "out2_weighted_mae")
        self.assertEqual(weighted_mae_metric.result().numpy(), 2.0)

        metric_container.reset_state()
        self.assertEqual(mse_metric.result().numpy(), 0.0)
        self.assertEqual(weighted_mse_metric.result().numpy(), 0.0)
        self.assertEqual(mae_metric.result().numpy(), 0.0)
        self.assertEqual(weighted_mae_metric.result().numpy(), 0.0)

    def test_metric_partial_dict_with_output_names(self):
        metric_container = compile_utils.MetricsContainer(
            {"out2": "mae"}, output_names=["out1", "out2"]
        )

        y_t = [tf.ones((10, 1)), tf.zeros((10, 1))]
        y_p = [tf.ones((10, 1)), tf.ones((10, 1))]
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        metric_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metric_container.metrics, 1)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "out2_mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

    def test_metric_partial_dict_with_nones(self):
        metric_container = compile_utils.MetricsContainer(
            {"out1": None, "out2": "mae"}
        )

        y_t = {"out1": tf.ones((10, 1)), "out2": tf.zeros((10, 1))}
        y_p = {"out1": tf.ones((10, 1)), "out2": tf.ones((10, 1))}
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        metric_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metric_container.metrics, 1)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "out2_mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

    def test_nested_structure(self):
        metric_container = compile_utils.MetricsContainer(
            metrics={"b": ["mse", None], "a": "mae"},
            weighted_metrics={"b": [None, None], "a": "mse"},
        )

        y_t = {
            "b": [2 * tf.ones((10, 1)), tf.zeros((10, 1))],
            "a": tf.zeros((10, 1)),
        }
        y_p = {
            "b": [tf.zeros((10, 1)), tf.zeros((10, 1))],
            "a": tf.ones((10, 1)),
        }
        sw = tf.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        metric_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metric_container.metrics, 3)

        a_mae_metric = metric_container.metrics[0]
        self.assertEqual(a_mae_metric.name, "a_mae")
        self.assertEqual(a_mae_metric.result().numpy(), 1.0)

        weighted_a_mae_metric = metric_container.metrics[1]
        self.assertEqual(weighted_a_mae_metric.name, "a_mse")
        self.assertEqual(weighted_a_mae_metric.result().numpy(), 1.0)

        b_1_mse_metric = metric_container.metrics[2]
        self.assertEqual(b_1_mse_metric.name, "b_1_mse")
        self.assertEqual(b_1_mse_metric.result().numpy(), 4.0)

    def test_no_input_mutation(self):
        metric = {"a": "mae"}
        metric_container = compile_utils.MetricsContainer(metric)

        y_t = {"a": tf.zeros((10, 1))}
        y_p = {"a": tf.ones((10, 1)), "b": tf.zeros((10, 1))}

        metric_container.update_state(y_t, y_p)
        self.assertLen(metric, 1)
        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.result().numpy(), 1.0)

    def test_crossentropy(self):
        metric_container = compile_utils.MetricsContainer("crossentropy")
        y_t, y_p = tf.ones((10, 1)), tf.ones((10, 1))
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn, metrics_mod.binary_crossentropy
        )

        metric_container = compile_utils.MetricsContainer("crossentropy")
        y_t, y_p = tf.ones((10, 1)), tf.ones((10, 20))
        self.assertEqual(y_p.shape.as_list()[-1], 20)
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn,
            metrics_mod.sparse_categorical_crossentropy,
        )

        metric_container = compile_utils.MetricsContainer("crossentropy")
        y_t, y_p = tf.ones((10, 20)), tf.ones((10, 20))
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn,
            metrics_mod.categorical_crossentropy,
        )

    def test_accuracy(self):
        metric_container = compile_utils.MetricsContainer("accuracy")
        y_t, y_p = tf.ones((10, 1)), tf.ones((10, 1))
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn, metrics_mod.binary_accuracy
        )

        metric_container = compile_utils.MetricsContainer("Accuracy")
        y_t, y_p = tf.ones((10, 1)), tf.ones((10, 1))
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn, metrics_mod.binary_accuracy
        )

        metric_container = compile_utils.MetricsContainer("accuracy")
        y_t, y_p = tf.ones((10, 1)), tf.ones((10, 20))
        self.assertEqual(y_p.shape.as_list()[-1], 20)
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn,
            metrics_mod.sparse_categorical_accuracy,
        )

        metric_container = compile_utils.MetricsContainer("accuracy")
        y_t, y_p = tf.ones((10, 20)), tf.ones((10, 20))
        metric_container.update_state(y_t, y_p)
        self.assertEqual(
            metric_container.metrics[0]._fn, metrics_mod.categorical_accuracy
        )

    def test_metric_weighting(self):
        metric_container = compile_utils.MetricsContainer(
            metrics=["mae"], weighted_metrics=["mae"]
        )

        y_t = tf.convert_to_tensor([[0], [3], [0]])
        y_p = tf.convert_to_tensor([[0], [0], [0]])
        sw = tf.convert_to_tensor([[1], [0], [1]])

        metric_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metric_container.metrics, 2)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

        weighted_mae_metric = metric_container.metrics[1]
        self.assertEqual(weighted_mae_metric.name, "weighted_mae")
        self.assertEqual(weighted_mae_metric.result().numpy(), 0.0)

    def test_broadcast_metrics_to_dict(self):
        metric_container = compile_utils.MetricsContainer(metrics=["mae"])

        y_p = {"output": tf.convert_to_tensor([[0], [1], [2]])}
        y_t = {"output": tf.convert_to_tensor([[1], [2], [3]])}
        metric_container.update_state(y_t, y_p)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

    def test_broadcast_metrics_to_dict_with_output_names(self):
        metric_container = compile_utils.MetricsContainer(
            metrics=["mae"], output_names=["output"]
        )

        y_p = tf.convert_to_tensor([[0], [1], [2]])
        y_t = {"output": tf.convert_to_tensor([[1], [2], [3]])}
        metric_container.update_state(y_t, y_p)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

    def test_missing_label_with_no_metrics(self):
        # It's ok to exclude a label if that label has no
        # losses or metrics associated with it.
        metric_container = compile_utils.MetricsContainer(
            metrics={"output1": "mae", "output3": "mse"}
        )

        y_p = {
            "output1": tf.convert_to_tensor([[0], [1], [2]]),
            "output2": tf.convert_to_tensor([[3], [4], [5]]),
            "output3": tf.convert_to_tensor([[6], [7], [8]]),
        }
        y_t = {
            "output1": tf.convert_to_tensor([[1], [2], [3]]),
            "output3": tf.convert_to_tensor([[4], [5], [6]]),
        }

        metric_container.update_state(y_t, y_p)
        self.assertLen(metric_container.metrics, 2)

        mae_metric = metric_container.metrics[0]
        self.assertEqual(mae_metric.name, "output1_mae")
        self.assertEqual(mae_metric.result().numpy(), 1.0)

        mse_metric = metric_container.metrics[1]
        self.assertEqual(mse_metric.name, "output3_mse")
        self.assertEqual(mse_metric.result().numpy(), 4.0)

    def test_metrics_masking(self):
        metrics_container = compile_utils.MetricsContainer(
            metrics=["mae"], weighted_metrics=["mse"]
        )
        y_p = tf.constant([[[1], [1]], [[0], [0]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        y_p._keras_mask = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)

        metrics_container.update_state(y_t, y_p)
        self.assertLen(metrics_container.metrics, 2)

        mae_metric = metrics_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertAlmostEqual(mae_metric.result().numpy(), 0)

        weighted_mae_metric = metrics_container.metrics[1]
        self.assertEqual(weighted_mae_metric.name, "mse")
        self.assertAlmostEqual(weighted_mae_metric.result().numpy(), 0)

    def test_metrics_sample_weight(self):
        metrics_container = compile_utils.MetricsContainer(
            metrics=["mae"], weighted_metrics=["mse"]
        )
        y_p = tf.constant([[[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        sw = tf.constant([[0.2, 0.3], [0.5, 0]], dtype=tf.float32)

        metrics_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metrics_container.metrics, 2)

        mae_metric = metrics_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertAlmostEqual(mae_metric.result().numpy(), 0.25)  # 1 / 4

        weighted_mae_metric = metrics_container.metrics[1]
        self.assertEqual(weighted_mae_metric.name, "mse")
        self.assertAlmostEqual(
            weighted_mae_metric.result().numpy(), 0.5
        )  # .5 / 1

    def test_metrics_masking_sample_weight(self):
        metrics_container = compile_utils.MetricsContainer(
            metrics=["mae"], weighted_metrics=["mse"]
        )
        y_p = tf.constant([[[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_t = tf.constant([[[1], [1]], [[1], [1]]], dtype=tf.float32)
        sw = tf.constant([[0.3, 0.2], [0.2, 0.3]], dtype=tf.float32)
        y_p._keras_mask = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)

        metrics_container.update_state(y_t, y_p, sample_weight=sw)
        self.assertLen(metrics_container.metrics, 2)

        mae_metric = metrics_container.metrics[0]
        self.assertEqual(mae_metric.name, "mae")
        self.assertAlmostEqual(mae_metric.result().numpy(), 0.5)  # 1 / .5

        weighted_mae_metric = metrics_container.metrics[1]
        self.assertEqual(weighted_mae_metric.name, "mse")
        self.assertAlmostEqual(weighted_mae_metric.result().numpy(), 0.2 / 0.5)

    def test_loss_class_as_metric_with_distribution(self):
        distribution = tf.distribute.OneDeviceStrategy("/device:CPU:0")
        with distribution.scope():
            metric_container = compile_utils.MetricsContainer(
                losses_mod.MeanSquaredError()
            )
            y_t, y_p = tf.ones((10, 5)), tf.zeros((10, 5))
            metric_container.update_state(y_t, y_p)

            self.assertLen(metric_container.metrics, 1)
            metric = metric_container.metrics[0]
            self.assertEqual(metric.name, "mean_squared_error")
            self.assertEqual(metric.result().numpy(), 1.0)

    def test_custom_metric_callables(self):
        def custom_metric_fn(y_true, y_pred):
            return tf.reduce_sum(y_true - y_pred)

        class CustomMetricClass:
            def __call__(self, y_true, y_pred):
                return tf.reduce_sum(y_true - y_pred)

        metric_container = compile_utils.MetricsContainer(
            [custom_metric_fn, CustomMetricClass()]
        )
        y_t, y_p = tf.ones((10, 5)), tf.zeros((10, 5))
        metric_container.update_state(y_t, y_p)

        self.assertEqual(metric_container.metrics[0].name, "custom_metric_fn")
        self.assertEqual(
            metric_container.metrics[1].name, "custom_metric_class"
        )

    def test_reset_state_existing_metric_before_built(self):
        metric = metrics_mod.Mean()
        metric.update_state([2.0, 4.0])
        self.assertEqual(metric.result().numpy(), 3.0)

        metric_container = compile_utils.MetricsContainer(metric)
        metric_container.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)

    def test_duplicated_metric_instance(self):
        mean_obj = metrics_mod.Mean()
        metric = mean_obj
        with self.assertRaisesRegex(ValueError, "Found duplicated metrics"):
            compile_utils.MetricsContainer(
                metrics=metric, weighted_metrics=metric
            )

        # duplicated string should be fine
        metric = "acc"
        compile_utils.MetricsContainer(metrics=metric, weighted_metrics=metric)

        # complicated structure
        metric = [mean_obj, "acc"]
        weighted_metric = {"output1": mean_obj, "output2": "acc"}
        with self.assertRaisesRegex(ValueError, "Found duplicated metrics"):
            compile_utils.MetricsContainer(
                metrics=metric, weighted_metrics=weighted_metric
            )


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
