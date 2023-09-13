# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dataset_creator."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.distribute import multi_worker_testing_utils
from keras.engine import data_adapter
from keras.engine import sequential
from keras.layers import core as core_layers
from keras.optimizers.legacy import gradient_descent
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import dataset_creator

# isort: off
from tensorflow.python.distribute.cluster_resolver import (
    SimpleClusterResolver,
)
from tensorflow.python.training.server_lib import (
    ClusterSpec,
)


@test_utils.run_v2_only
class DatasetCreatorTest(tf.test.TestCase, parameterized.TestCase):
    def test_dataset_creator(self):
        with self.assertRaisesRegex(
            TypeError, "`dataset_fn` for `DatasetCreator` must be a `callable`."
        ):
            dataset_creator.DatasetCreator(2)

        dataset_fn = lambda: 3
        with self.assertRaisesRegex(
            TypeError,
            "The `callable` provided to `DatasetCreator` must return "
            "a Dataset.",
        ):
            dataset_creator.DatasetCreator(dataset_fn)()

        dataset_fn = lambda: tf.data.Dataset.from_tensor_slices([1, 1])
        got = dataset_creator.DatasetCreator(dataset_fn)()
        self.assertEqual(
            next(iter(got)),
            next(iter(tf.data.Dataset.from_tensor_slices([1, 1]))),
        )

    def _get_dataset_fn(self):
        def dataset_fn(input_context):
            global_batch_size = 64
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size
            )
            dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat()
            dataset = dataset.shard(
                input_context.num_input_pipelines,
                input_context.input_pipeline_id,
            )
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2)
            return dataset

        return dataset_fn

    @test_combinations.generate(
        test_combinations.combine(use_input_options=[True, False])
    )
    def test_dataset_creator_model_fit_without_strategy(
        self, use_input_options
    ):
        model = sequential.Sequential([core_layers.Dense(10)])
        model.compile(gradient_descent.SGD(), loss="mse")

        input_options = (
            tf.distribute.InputOptions() if use_input_options else None
        )
        history = model.fit(
            dataset_creator.DatasetCreator(
                self._get_dataset_fn(), input_options
            ),
            epochs=10,
            steps_per_epoch=10,
            verbose=0,
        )
        self.assertLen(history.history["loss"], 10)

    def _get_parameter_server_strategy(self):
        cluster_def = multi_worker_testing_utils.create_in_process_cluster(
            num_workers=2, num_ps=1, rpc_layer="grpc"
        )
        return tf.distribute.experimental.ParameterServerStrategy(
            SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer="grpc")
        )

    @test_combinations.generate(
        test_combinations.combine(use_input_options=[True, False])
    )
    def test_dataset_creator_usage_in_parameter_server_model_fit(
        self, use_input_options
    ):
        strategy = self._get_parameter_server_strategy()
        with strategy.scope():
            model = sequential.Sequential([core_layers.Dense(10)])
        model.compile(gradient_descent.SGD(), loss="mse")

        input_options = (
            tf.distribute.InputOptions() if use_input_options else None
        )
        history = model.fit(
            dataset_creator.DatasetCreator(
                self._get_dataset_fn(), input_options
            ),
            epochs=10,
            steps_per_epoch=10,
            verbose=0,
        )
        self.assertLen(history.history["loss"], 10)

    def test_dataset_creator_input_options(self):
        dataset_fn = lambda _: tf.data.Dataset.from_tensor_slices([1, 1])
        input_options = tf.distribute.InputOptions(
            experimental_fetch_to_device=True,
            experimental_per_replica_buffer_size=2,
        )
        x = dataset_creator.DatasetCreator(
            dataset_fn, input_options=input_options
        )
        with tf.distribute.MultiWorkerMirroredStrategy().scope():
            data_handler = data_adapter.get_data_handler(
                x,
                steps_per_epoch=2,
                model=sequential.Sequential([core_layers.Dense(10)]),
            )

        # Ensuring the resulting `DistributedDatasetsFromFunction` has the right
        # options.
        self.assertTrue(
            data_handler._dataset._options.experimental_fetch_to_device
        )
        self.assertEqual(
            data_handler._dataset._options.experimental_per_replica_buffer_size,
            2,
        )

    def test_dataset_creator_input_options_with_cluster_coordinator(self):
        dataset_fn = lambda _: tf.data.Dataset.from_tensor_slices([1, 1])
        input_options = tf.distribute.InputOptions(
            experimental_fetch_to_device=True,
            experimental_per_replica_buffer_size=2,
        )
        x = dataset_creator.DatasetCreator(
            dataset_fn, input_options=input_options
        )
        strategy = self._get_parameter_server_strategy()
        with strategy.scope():
            model = sequential.Sequential([core_layers.Dense(10)])
            model._cluster_coordinator = (
                tf.distribute.experimental.coordinator.ClusterCoordinator(
                    strategy
                )
            )
            data_handler = data_adapter.get_data_handler(
                x, steps_per_epoch=2, model=model
            )

        iter_rv = iter(data_handler._dataset)._values[0]
        iter_rv._rebuild_on(model._cluster_coordinator._cluster.workers[0])
        distributed_iterator = iter_rv._get_values()

        # Ensuring the resulting `DistributedIterator` has the right options.
        self.assertTrue(
            distributed_iterator._options.experimental_fetch_to_device
        )
        self.assertEqual(
            distributed_iterator._options.experimental_per_replica_buffer_size,
            2,
        )


if __name__ == "__main__":
    tf.test.main()
