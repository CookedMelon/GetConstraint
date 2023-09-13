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
"""Custom Training Loop correctness test."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import optimizers
from keras.applications import resnet_v2
from keras.datasets import fashion_mnist
from keras.distribute import optimizer_combinations
from keras.distribute import strategy_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.ops.losses import losses_impl

_NUM_SAMPLES = 66
_BATCH_SIZE = 32
_RANDOM_SEED = 1337
_NUM_EPOCHS = 2
_STEPS_PER_EPOCH = 2


class MaybeStrategyScope:
    """Provides a context allowing no distribution strategy."""

    def __init__(self, strategy):
        self._strategy = strategy
        self._scope = None

    def __enter__(self):
        if self._strategy:
            self._scope = self._strategy.scope()
            self._scope.__enter__()

    def __exit__(self, exc_type, value, traceback):
        if self._strategy:
            self._scope.__exit__(exc_type, value, traceback)
            self._scope = None


def get_model(sync_batchnorm=False):
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation="relu", input_shape=(1,)))
    model.add(
        keras.layers.Dense(
            10,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
    )
    if sync_batchnorm:
        model.add(keras.layers.BatchNormalization(synchronized=True))
    else:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(1))
    return model


def get_data():
    x_train = np.random.rand(_NUM_SAMPLES, 1)
    y_train = 3 * x_train
    x_train = x_train.astype("float32")
    y_train = y_train.astype("float32")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(_BATCH_SIZE)
    return train_dataset


def compute_loss(labels, logits, reg_losses):
    pred_loss = keras.losses.mean_squared_error(labels, logits)
    scaled_loss = tf.nn.compute_average_loss(
        pred_loss, global_batch_size=_BATCH_SIZE
    )
    l2_loss = tf.nn.scale_regularization_loss(reg_losses)
    return scaled_loss + l2_loss


def iteration_inside_func(
    initial_weights,
    dataset,
    optimizer_fn,
    iteration_type,
    strategy=None,
    sync_batchnorm=None,
    jit_compile=False,
):
    """Helper function to test iterating over data inside a tf.function."""
    with MaybeStrategyScope(strategy):
        if strategy and sync_batchnorm:
            model = get_model(sync_batchnorm)
        else:
            model = get_model()
        model.set_weights(initial_weights)
        optimizer = optimizer_fn()

        training_accuracy = keras.metrics.CategoricalAccuracy(
            "training_accuracy", dtype=tf.float32
        )

        @tf.function
        def train_epoch(dist_input):
            """Training StepFn."""

            @tf.function(jit_compile=jit_compile)
            def step_fn(inputs):
                samples, labels = inputs
                with tf.GradientTape() as tape:
                    logits = model(samples)
                    loss = compute_loss(labels, logits, model.losses)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                training_accuracy.update_state(labels, logits)
                return loss

            total_loss = 0.0
            num_batches = 0
            if iteration_type == "dataset":
                for x in dist_input:
                    if strategy:
                        per_replica_losses = strategy.run(step_fn, args=(x,))
                        total_loss += strategy.reduce(
                            tf.distribute.ReduceOp.SUM,
                            per_replica_losses,
                            axis=None,
                        )
                    else:
                        total_loss += step_fn(x)
                    num_batches += 1
            else:
                iterator = iter(dist_input)
                for _ in range(_STEPS_PER_EPOCH):
                    if strategy:
                        per_replica_losses = strategy.run(
                            step_fn, args=(next(iterator),)
                        )
                        total_loss += strategy.reduce(
                            tf.distribute.ReduceOp.SUM,
                            per_replica_losses,
                            axis=None,
                        )
                    else:
                        total_loss += step_fn(next(iterator))
                    num_batches += 1

            return total_loss / tf.cast(num_batches, dtype=tf.float32)

        if strategy:
            dataset = strategy.experimental_distribute_dataset(dataset)

        for _ in range(_NUM_EPOCHS):
            loss = train_epoch(dataset)

        return (model.get_weights(), loss, training_accuracy.result())


def iteration_outside_func(
    initial_weights,
    dataset,
    optimizer_fn,
    iteration_type,
    strategy=None,
    sync_batchnorm=None,
    jit_compile=False,
):
    """Helper function to test iterating over data outside a tf.function."""
    with MaybeStrategyScope(strategy):
        model = get_model(sync_batchnorm=sync_batchnorm)
        model.set_weights(initial_weights)
        optimizer = optimizer_fn()

        training_accuracy = keras.metrics.CategoricalAccuracy(
            "training_accuracy", dtype=tf.float32
        )

        @tf.function
        def train_step(dist_inputs):
            """Training StepFn."""

            @tf.function(jit_compile=jit_compile)
            def step_fn(inputs):
                samples, labels = inputs
                with tf.GradientTape() as tape:
                    logits = model(samples)
                    loss = compute_loss(labels, logits, model.losses)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                training_accuracy.update_state(labels, logits)
                return loss

            if strategy:
                per_replica_losses = strategy.run(step_fn, args=(dist_inputs,))
                return strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
                )
            else:
                return step_fn(dist_inputs)

        if strategy:
            dataset = strategy.experimental_distribute_dataset(dataset)

        total_loss = 0.0
        num_batches = 0
        if iteration_type == "dataset":
            for _ in range(_NUM_EPOCHS):
                for x in dataset:
                    total_loss += train_step(x)
                    num_batches += 1
        else:
            for _ in range(_NUM_EPOCHS):
                iterator = iter(dataset)
                for _ in range(_STEPS_PER_EPOCH):
                    total_loss += train_step(next(iterator))
                    num_batches += 1

        return (
            model.get_weights(),
            total_loss / tf.cast(num_batches, dtype=tf.float32),
            training_accuracy.result(),
        )


@test_utils.run_v2_only
class TestDistributionStrategyDnnCorrectness(
    tf.test.TestCase, parameterized.TestCase
):
    """Test custom training loop correctness with a simple DNN model."""

    def setUp(self):
        super().setUp()
        np.random.seed(_RANDOM_SEED)
        tf.compat.v1.set_random_seed(_RANDOM_SEED)

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            distribution=strategy_combinations.all_strategies,
            optimizer_fn=optimizer_combinations.optimizers_v2,
            mode=["eager"],
            iteration_type=["iterator", "dataset"],
            inside_func=[False, True],
            sync_batchnorm=[True, False],
            jit_compile=[False],
        )
        + tf.__internal__.test.combinations.combine(
            distribution=strategy_combinations.multiworker_strategies,
            optimizer_fn=[
                optimizer_combinations.gradient_descent_optimizer_keras_v2_fn,
                optimizer_combinations.adagrad_optimizer_keras_v2_fn,
                optimizer_combinations.adam_experimental_fn,
            ],
            mode=["eager"],
            iteration_type=["iterator", "dataset"],
            inside_func=[False, True],
            sync_batchnorm=[True, False],
            jit_compile=[False],
        )
        + tf.__internal__.test.combinations.combine(
            distribution=[
                tf.__internal__.distribute.combinations.one_device_strategy_gpu,
                tf.__internal__.distribute.combinations.mirrored_strategy_with_two_gpus,  # noqa: E501
            ],
            optimizer_fn=[
                optimizer_combinations.gradient_descent_optimizer_keras_v2_fn,
                optimizer_combinations.adagrad_optimizer_keras_v2_fn,
            ],
            mode=["eager"],
            iteration_type=["iterator", "dataset"],
            inside_func=[False, True],
            sync_batchnorm=[True, False],
            jit_compile=[True],
        )
    )
    def test_dnn_correctness_minus_tpus(
        self,
        distribution,
        optimizer_fn,
        iteration_type,
        inside_func,
        sync_batchnorm,
        jit_compile,
    ):
        # TODO(anjs): Identify why this particular V1 optimizer needs a higher
        # tol.
        if (
            "FtrlV1" in optimizer_fn._name
            and "TPU" in type(distribution).__name__
        ):
            self.skipTest("Reduced tolerance of the order of 1e-1 required.")
        self.dnn_correctness(
            distribution,
            optimizer_fn,
            iteration_type,
            inside_func,
            sync_batchnorm,
            jit_compile,
        )

    def dnn_correctness(
        self,
        distribution,
        optimizer_fn,
        iteration_type,
        inside_func,
        sync_batchnorm=None,
        jit_compile=False,
    ):
        model = get_model(sync_batchnorm)
        initial_weights = model.get_weights()
        dataset = get_data()
        if inside_func:
            iteration_func = iteration_inside_func
        else:
            iteration_func = iteration_outside_func

        wts_with_ds, loss_with_ds, acc_with_ds = iteration_func(
            initial_weights,
            dataset,
            optimizer_fn,
            iteration_type,
            strategy=distribution,
            sync_batchnorm=sync_batchnorm,
            jit_compile=jit_compile,
        )
        wts, loss, acc = iteration_func(
            initial_weights,
            dataset,
            optimizer_fn,
            iteration_type,
            sync_batchnorm=sync_batchnorm,
            jit_compile=False,
        )

        self.assertAllClose(wts, wts_with_ds, atol=1e-3, rtol=1e-3)
        self.assertAllClose(loss, loss_with_ds, atol=1e-3, rtol=1e-3)
        self.assertAllClose(acc, acc_with_ds, atol=1e-3, rtol=1e-3)

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            distribution=[
                tf.__internal__.distribute.combinations.mirrored_strategy_with_two_gpus,  # noqa: E501
            ],
            mode=["eager"],
        )
    )
    def test_fused_batch_norm_uneven_batch(self, distribution):
        """Test that fused BN works when the last device gets empty data.

        Adapted from
        https://www.tensorflow.org/tutorials/distribute/custom_training
        but using ResNet, which uses fused batchnorm, as the model.

        Arguments:
          distribution: distribute test configuration
        """
        self.skipTest("TODO(b/234354008): Requires fetching data from network.")
        (train_images, train_labels), _ = fashion_mnist.load_data()
        # add channel dimension to make 2D data into 3D, since some ops of the
        # model require it.
        train_images = train_images[..., None]
        train_images = train_images / np.float32(255)

        # Padding images because ResNet requires a minimal shape of (32, 32)
        padded_train_images = np.concatenate(
            [
                np.zeros((len(train_images), 2, 28, 1)),
                train_images,
                np.zeros((len(train_images), 2, 28, 1)),
            ],
            axis=1,
        )
        padded_train_images = np.concatenate(
            [
                np.zeros((len(train_images), 32, 2, 1)),
                padded_train_images,
                np.zeros((len(train_images), 32, 2, 1)),
            ],
            axis=2,
        )

        buffer_size = len(train_images)
        global_batch_size = distribution.num_replicas_in_sync
        num_samples = global_batch_size - 1

        epochs = 2

        # Keep only the first images, so that the last GPU receives an empty
        # batch
        padded_train_images = padded_train_images[:num_samples]
        train_labels = train_labels[:num_samples]

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (padded_train_images, train_labels)
            )
            .shuffle(buffer_size)
            .batch(global_batch_size)
        )
        train_dist_dataset = distribution.experimental_distribute_dataset(
            train_dataset
        )

        def create_model():
            inputs = keras.Input((32, 32, 1))
            preprocessed = keras.layers.Conv2D(3, (1, 1))(
                inputs
            )  # ResNet requires 3 channels
            features = resnet_v2.ResNet50V2(
                include_top=False,
                input_tensor=preprocessed,
                pooling="avg",
                weights=None,
            ).output
            return keras.Model(inputs, features)

        with distribution.scope():
            # Set reduction to `none` so we can do the reduction afterwards and
            # divide by global batch size.
            loss_object = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=losses_impl.Reduction.NONE
            )

            def compute_resnet_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=global_batch_size
                )

            model = create_model()

            optimizer = optimizers.adam_legacy.Adam()

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_resnet_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = distribution.run(
                train_step, args=(dataset_inputs,)
            )
            return distribution.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
            )

        for epoch in range(epochs):
            # Train loop
            total_loss = 0.0
            num_batches = 0
            for x in train_dist_dataset:
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            print(f"Epoch {epoch+1}, Loss: {train_loss}")


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
