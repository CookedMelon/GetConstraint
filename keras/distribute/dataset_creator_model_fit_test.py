# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `DatasetCreator` with `Model.fit` across usages and strategies."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras.distribute import dataset_creator_model_fit_test_base as test_base
from keras.distribute import strategy_combinations
from keras.testing_infra import test_utils
from keras.utils import dataset_creator

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


# TODO(rchao): Investigate why there cannot be single worker and multi worker
# PS strategies running in the same shard.
@test_utils.run_v2_only
@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies
        + strategy_combinations.multi_worker_mirrored_strategies
        + strategy_combinations.parameter_server_strategies_multi_worker,
        mode="eager",
    )
)
class DatasetCreatorModelFitTest(test_base.DatasetCreatorModelFitTestBase):
    def setUp(self):
        super().setUp()
        if tf_test_utils.is_xla_enabled():
            self.skipTest(
                "model.optimizer.iterations values is not as expected "
                "with XLA: b/184384487"
            )

    def testModelFit(self, strategy):
        model = self._model_fit(strategy)
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitwithStepsPerEpochNegativeOne(self, strategy):
        def dataset_fn(input_context):
            del input_context
            x = tf.random.uniform((10, 10))
            y = tf.random.uniform((10,))
            return (
                tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).batch(2)
            )

        if strategy._should_use_with_coordinator:
            with self.assertRaises(
                (tf.errors.OutOfRangeError, tf.errors.CancelledError)
            ):
                self._model_fit(
                    strategy,
                    steps_per_epoch=-1,
                    x=dataset_creator.DatasetCreator(dataset_fn),
                    validation_data=dataset_creator.DatasetCreator(dataset_fn),
                )
        else:
            self._model_fit(
                strategy,
                steps_per_epoch=-1,
                x=dataset_creator.DatasetCreator(dataset_fn),
                validation_data=dataset_creator.DatasetCreator(dataset_fn),
            )

    def testModelFitWithNumpyData(self, strategy):
        x = np.random.rand(100, 10)
        y = np.random.rand(100, 1)
        model = self._model_fit(
            strategy,
            x=x,
            y=y,
            batch_size=1,
            validation_data=(x, y),
        )
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitWithTensorData(self, strategy):
        x = tf.random.uniform((100, 10))
        y = tf.random.uniform((100,))
        model = self._model_fit(
            strategy,
            x=x,
            y=y,
            batch_size=1,
            validation_data=(x, y),
        )
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitWithLookupLayer(self, strategy):
        model = self._model_fit(strategy, use_lookup_layer=True)
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitWithNormalizationLayer(self, strategy):
        model = self._model_fit(strategy, with_normalization_layer=True)
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitWithStepsPerExecution(self, strategy):
        model = self._model_fit(strategy, steps_per_execution=10)
        self.assertEqual(model.optimizer.iterations, 100)

    def testModelFitWithNoStepsPerEpoch(self, strategy):
        with self.assertRaisesRegex(
            ValueError,
            "When using a `tf.keras.utils.experimental.DatasetCreator`, "
            "`steps_per_epoch`, `validation_steps`, `steps`, or "
            "`pss_evaluation_shards` argument must be provided in "
            "`Model.fit`, `Model.evaluate`, or `Model.predict`.",
        ):
            self._model_fit(strategy, steps_per_epoch=None)

    def testModelEvaluate(self, strategy):
        self._model_evaluate(strategy)
        self.assertGreaterEqual(self._accuracy_metric.result(), 0.0)

    def testModelEvaluateWithNumpyData(self, strategy):
        x = np.random.rand(100, 10)
        y = np.random.rand(100, 1)
        self._model_evaluate(
            strategy,
            x=x,
            y=y,
            batch_size=1,
        )
        self.assertGreaterEqual(self._accuracy_metric.result(), 0.0)

    def testModelEvaluateWithTensorData(self, strategy):
        x = tf.random.uniform((100, 10))
        y = tf.random.uniform((100,))
        self._model_evaluate(
            strategy,
            x=x,
            y=y,
            batch_size=1,
        )
        self.assertGreaterEqual(self._accuracy_metric.result(), 0.0)

    def testModelEvaluateWithNormalizationLayer(self, strategy):
        self._model_evaluate(strategy, with_normalization_layer=True)
        self.assertGreaterEqual(self._accuracy_metric.result(), 0.0)

    def testModelEvaluateWithStepsPerExecution(self, strategy):
        self._model_evaluate(strategy, steps_per_execution=10)
        self.assertGreaterEqual(self._accuracy_metric.result(), 0.0)

    def testModelEvaluateWithNoStepsPerEpoch(self, strategy):
        with self.assertRaisesRegex(
            ValueError,
            "When using a `tf.keras.utils.experimental.DatasetCreator`, "
            "`steps_per_epoch`, `validation_steps`, `steps`, or "
            "`pss_evaluation_shards` argument must be provided in "
            "`Model.fit`, `Model.evaluate`, or `Model.predict`.",
        ):
            self._model_evaluate(strategy, steps=None)

    def testModelPredict(self, strategy):
        _, predictions = self._model_predict(strategy, steps=3)
        # Check the first (0th index), fourth (3rd index) and the last
        # predictions because the first, fourth and the last input are the same
        # in `model.predict` so there predictions should match.
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )

        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelPredictWithNumpyData(self, strategy):
        x = np.array([[1.0], [2.0], [3.0], [1.0], [5.0], [1.0]])
        _, predictions = self._model_predict(strategy, test_data=x)

        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )
        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelPredictWithTensorData(self, strategy):
        x = tf.constant([[1.0], [2.0], [3.0], [1.0], [5.0], [1.0]])
        _, predictions = self._model_predict(strategy, test_data=x)
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )
        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelPredictWithNormalizationLayer(self, strategy):
        _, predictions = self._model_predict(
            strategy, with_normalization_layer=True, steps=3
        )
        # Check the first (0th index), fourth (3rd index) and the last
        # predictions because the first, fourth and the last input is the same
        # in `model.predict` so there predictions should match.
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )

        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelPredictWithStepsPerExecution(self, strategy):
        _, predictions = self._model_predict(
            strategy, steps_per_execution=3, steps=3
        )

        # Check the first (0th index), fourth (3rd index) and the last
        # predictions because the first, fourth and the last input is the same
        # in `model.predict` so there predictions should match.
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )

        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelFitAndPredict(self, strategy):
        def fit_dataset_fn(input_context):
            del input_context
            x = tf.random.uniform((10, 1))
            y = tf.random.uniform((10,))
            return (
                tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(10)
                .repeat()
                .batch(2)
            )

        x = dataset_creator.DatasetCreator(fit_dataset_fn)
        validation_data = dataset_creator.DatasetCreator(fit_dataset_fn)

        model = self._model_fit(strategy, x=x, validation_data=validation_data)
        _, predictions = self._model_predict(strategy, model, steps=3)

        # Check the first (0th index), fourth (3rd index) and the last
        # predictions because the first, fourth and the last input is the same
        # in `model.predict` so there predictions should match.
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )

        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelPredictWithDatasetCreator(self, strategy):
        if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
            self.skipTest("b/189223991")

        def _dataset_fn(input_context):
            del input_context
            x = tf.constant([[1.0], [2.0], [3.0], [1.0], [5.0], [1.0]])
            return tf.data.Dataset.from_tensor_slices(x).repeat().batch(2)

        _, predictions = self._model_predict(
            strategy,
            steps=3,
            test_data=dataset_creator.DatasetCreator(_dataset_fn),
        )

        # Check the first (0th index), fourth (3rd index) and the last
        # predictions because the first, fourth and the last input is the same
        # in `model.predict` so there predictions should match.
        self.assertTrue(
            all(predictions[0] == predictions[i] for i in [0, 3, 5])
        )

        self.assertFalse(
            all(predictions[0] == predictions[i] for i in [0, 1, 2, 4])
        )

    def testModelTrainTFFunction(self, strategy):
        model = self._model_fit(strategy)
        self.assertIsInstance(
            model.train_tf_function, tf.__internal__.function.Function
        )


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
