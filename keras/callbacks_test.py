# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras callbacks."""


import collections
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.callbacks import BackupAndRestore
from keras.callbacks import BackupAndRestoreExperimental
from keras.callbacks import Callback
from keras.engine import sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import sgd
from keras.optimizers.legacy import gradient_descent
from keras.optimizers.schedules import learning_rate_schedule
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import io_utils
from keras.utils import np_utils
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging

try:
    import h5py
except ImportError:
    h5py = None

try:
    import requests
except ImportError:
    requests = None


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5

CALLBACK_HOOKS = [
    "on_batch_begin",
    "on_batch_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_predict_batch_begin",
    "on_predict_batch_end",
    "on_predict_begin",
    "on_predict_end",
    "on_test_batch_begin",
    "on_test_batch_end",
    "on_test_begin",
    "on_test_end",
    "on_train_batch_begin",
    "on_train_batch_end",
    "on_train_begin",
    "on_train_end",
]


class Counter(keras.callbacks.Callback):
    """Counts the number of times each callback method was run.

    Attributes:
      method_counts: dict. Contains the counts of time  each callback method was
        run.
    """

    def __init__(self):
        self.method_counts = collections.defaultdict(int)
        for method_name in CALLBACK_HOOKS:
            setattr(
                self,
                method_name,
                self.wrap_with_counts(method_name, getattr(self, method_name)),
            )

    def wrap_with_counts(self, method_name, method):
        def _call_and_count(*args, **kwargs):
            self.method_counts[method_name] += 1
            return method(*args, **kwargs)

        return _call_and_count


class CallAllHooks(keras.callbacks.Callback):
    """A callback that calls self._run for all hooks"""

    def __init__(self):
        for method_name in CALLBACK_HOOKS:
            setattr(self, method_name, self._run)

    def _run(self, *args, logs=None):
        raise NotImplementedError


def _get_numpy():
    return np.ones((10, 10)), np.ones((10, 1))


def _get_sequence():
    class MySequence(keras.utils.data_utils.Sequence):
        def __getitem__(self, _):
            return np.ones((2, 10)), np.ones((2, 1))

        def __len__(self):
            return 5

    return MySequence(), None


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class CallbackCountsTest(test_combinations.TestCase):
    def _check_counts(self, counter, expected_counts):
        """Checks that the counts registered by `counter` are those expected."""
        for method_name, expected_count in expected_counts.items():
            self.assertEqual(
                counter.method_counts[method_name],
                expected_count,
                msg="For method {}: expected {}, got: {}".format(
                    method_name,
                    expected_count,
                    counter.method_counts[method_name],
                ),
            )

    def _get_model(self):
        layers = [
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(10,))
        model.compile(
            tf.compat.v1.train.AdamOptimizer(0.001),
            "binary_crossentropy",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        return model

    @parameterized.named_parameters(
        ("with_numpy", _get_numpy()), ("with_sequence", _get_sequence())
    )
    def test_callback_hooks_are_called_in_fit(self, data):
        if not tf.executing_eagerly():
            self.skipTest("Behavior changed in v2.")
        x, y = data
        val_x, val_y = np.ones((4, 10)), np.ones((4, 1))

        model = self._get_model()
        counter = Counter()
        model.fit(
            x,
            y,
            validation_data=(val_x, val_y),
            batch_size=2,
            steps_per_epoch=5,
            epochs=5,
            callbacks=[counter],
        )

        self._check_counts(
            counter,
            {
                "on_batch_begin": 25,
                "on_batch_end": 25,
                "on_epoch_begin": 5,
                "on_epoch_end": 5,
                "on_predict_batch_begin": 0,
                "on_predict_batch_end": 0,
                "on_predict_begin": 0,
                "on_predict_end": 0,
                "on_test_batch_begin": 10,
                "on_test_batch_end": 10,
                "on_test_begin": 5,
                "on_test_end": 5,
                "on_train_batch_begin": 25,
                "on_train_batch_end": 25,
                "on_train_begin": 1,
                "on_train_end": 1,
            },
        )

    @parameterized.named_parameters(
        ("with_numpy", _get_numpy()), ("with_sequence", _get_sequence())
    )
    def test_callback_hooks_are_called_in_evaluate(self, data):
        x, y = data
        is_sequence = isinstance(x, keras.utils.data_utils.Sequence)

        model = self._get_model()
        counter = Counter()
        model.evaluate(
            x,
            y,
            batch_size=2 if not is_sequence else None,
            steps=5 if is_sequence else None,
            callbacks=[counter],
        )
        self._check_counts(
            counter,
            {
                "on_test_batch_begin": 5,
                "on_test_batch_end": 5,
                "on_test_begin": 1,
                "on_test_end": 1,
            },
        )

    @parameterized.named_parameters(
        ("with_numpy", _get_numpy()), ("with_sequence", _get_sequence())
    )
    def test_callback_hooks_are_called_in_predict(self, data):
        x = data[0]
        is_sequence = isinstance(x, keras.utils.data_utils.Sequence)

        model = self._get_model()
        counter = Counter()
        model.predict(
            x,
            batch_size=2 if not is_sequence else None,
            steps=5 if is_sequence else None,
            callbacks=[counter],
        )
        self._check_counts(
            counter,
            {
                "on_predict_batch_begin": 5,
                "on_predict_batch_end": 5,
                "on_predict_begin": 1,
                "on_predict_end": 1,
            },
        )

    def test_callback_list_methods(self):
        counter = Counter()
        callback_list = keras.callbacks.CallbackList([counter])

        batch = 0
        callback_list.on_test_batch_begin(batch)
        callback_list.on_test_batch_end(batch)
        callback_list.on_predict_batch_begin(batch)
        callback_list.on_predict_batch_end(batch)

        self._check_counts(
            counter,
            {
                "on_test_batch_begin": 1,
                "on_test_batch_end": 1,
                "on_predict_batch_begin": 1,
                "on_predict_batch_end": 1,
            },
        )


class KerasCallbacksTest(test_combinations.TestCase):
    def _get_model(self, input_shape=None, additional_metrics=None):
        additional_metrics = additional_metrics or []
        layers = [
            keras.layers.Dense(3, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
        model = test_utils.get_model_from_layers(
            layers, input_shape=input_shape
        )
        model.compile(
            loss="mse",
            optimizer="rmsprop",
            metrics=[keras.metrics.CategoricalAccuracy(name="my_acc")]
            + additional_metrics,
            run_eagerly=test_utils.should_run_eagerly(),
        )
        return model

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_progbar_logging(self):
        model = self._get_model(input_shape=(3,))

        x = tf.ones((200, 3))
        y = tf.zeros((200, 2))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        expected_log = r"(.*- loss:.*- my_acc:.*)+"

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(dataset, epochs=2, steps_per_epoch=10)
            self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_progbar_logging_with_stateful_metrics(self):
        class AddAllOnes(keras.metrics.Metric):
            """A simple metric that adds all the one's in `y_true`."""

            def __init__(self, name="add_all_ones", **kwargs):
                super().__init__(name=name, **kwargs)
                self.total = self.add_weight(name="total", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.total.assign_add(
                    tf.cast(tf.reduce_sum(y_true), dtype=tf.float32)
                )

            def result(self):
                return self.total

        x_train = np.array([[0, 1, 0, 1, 0, 1, 0, 1]] * 8).astype(float)
        y_train = np.array(
            [[1, 0], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 0]]
        )
        # There are 7 ones in total in `y_train` after two batches.
        expected_log = r"(.*- loss:.*- my_acc:.*- add_all_ones: 7.0000)+"

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model = self._get_model(
                input_shape=(8,), additional_metrics=[AddAllOnes()]
            )
            model.fit(x_train, y_train, verbose=1, batch_size=4, shuffle=False)
            self.assertRegex(printed.contents(), expected_log)

        # When not executing eagerly, `model.evaluate` does not have the metrics
        # results printed.
        if tf.executing_eagerly():
            with self.captureWritesToStream(sys.stdout) as printed:
                model = self._get_model(
                    input_shape=(8,), additional_metrics=[AddAllOnes()]
                )
                model.evaluate(x_train, y_train, verbose=1, batch_size=4)
                self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_all_keras_modes
    def test_trivial_backup_restore(self):
        if test_utils.should_run_eagerly():
            model = keras.Sequential([keras.layers.Dense(1)])
            model.compile("sgd", "mse")
            cbk = BackupAndRestore(self.get_temp_dir())
            model.fit(
                np.ones((10, 1)), np.ones((10, 1)), epochs=1, callbacks=[cbk]
            )

    def test_backup_restore_train_counter(self):
        if not tf.compat.v1.executing_eagerly():
            self.skipTest(
                "BackupAndRestore only available when eager execution is "
                "enabled"
            )
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")
        cbk = BackupAndRestore(self.get_temp_dir())

        class InterruptingCallback(keras.callbacks.Callback):
            """A callback to intentionally introduce interruption to
            training."""

            def on_epoch_end(self, epoch, log=None):
                logging.info(f"counter: {model._train_counter}")
                if epoch == 5 or epoch == 12:
                    raise RuntimeError("Interruption")

        self.get_temp_dir()

        # The following asserts that the train counter is fault tolerant.
        self.assertEqual(model._train_counter.numpy(), 0)
        try:
            model.fit(
                np.ones((10, 1)),
                np.ones((10, 1)),
                epochs=20,
                callbacks=[cbk, InterruptingCallback()],
            )
        except RuntimeError:
            pass
        self.assertEqual(model._train_counter.numpy(), 6)
        try:
            model.fit(
                np.ones((10, 1)),
                np.ones((10, 1)),
                epochs=20,
                callbacks=[cbk, InterruptingCallback()],
            )
        except RuntimeError:
            pass
        self.assertEqual(model._train_counter.numpy(), 13)

    def _test_backup_and_restore_callback_with(self, cls):
        if not tf.compat.v1.executing_eagerly():
            self.skipTest(
                "BackupAndRestore only available when execution is enabled"
            )

        class InterruptingCallback(keras.callbacks.Callback):
            """A callback to intentionally introduce interruption to
            training."""

            def on_epoch_end(self, epoch, log=None):
                if epoch == 15:
                    raise RuntimeError("Interruption")

        model = keras.Sequential([keras.layers.Dense(10)])
        optimizer = sgd.SGD()
        model.compile(optimizer, loss="mse")

        x = tf.random.uniform((24, 10))
        y = tf.random.uniform((24,))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(2)

        backup_callback = cls(backup_dir=self.get_temp_dir())
        try:
            model.fit(
                dataset,
                epochs=20,
                steps_per_epoch=5,
                callbacks=[backup_callback, InterruptingCallback()],
            )
        except RuntimeError:
            logging.warning("***Handling interruption***")
            # This continues at the epoch where it left off.
            model.fit(
                dataset,
                epochs=20,
                steps_per_epoch=5,
                callbacks=[backup_callback],
            )

    def _test_backup_and_restore_callback_at_steps(
        self, cls, epoch_int, steps_int, mode
    ):
        if not tf.compat.v1.executing_eagerly():
            self.skipTest(
                "BackupAndRestore only available when eager execution is "
                "enabled"
            )

        class InterruptingCallback(keras.callbacks.Callback):
            """A callback to intentionally introduce interruption to
            training."""

            batch_count = 0

            def on_epoch_end(self, epoch, log=None):
                if epoch == epoch_int:
                    raise RuntimeError("EpochInterruption")

            def on_batch_end(self, batch, logs=None):
                self.batch_count += 1
                if self.batch_count == steps_int:
                    raise RuntimeError("StepsInterruption")

        class VerifyRestore(Callback):
            """Verify if the training restored to the correct epoch and step."""

            def __init__(self, initial_epoch, initial_step):
                super(VerifyRestore, self).__init__()
                self.initial_epoch = initial_epoch
                self.initial_step = initial_step
                self._current_epoch = 0

            def on_epoch_begin(self, epoch, logs=None):
                self._current_epoch = epoch
                if epoch < self.initial_epoch:
                    raise ValueError(
                        "Training did not restore at epoch (%d) and step (%d)"
                        % (self.initial_epoch, self.initial_step)
                    )

            def on_batch_begin(self, batch, logs=None):
                if (
                    batch <= self.initial_step
                    and self._current_epoch < self.initial_epoch
                ):
                    raise ValueError(
                        "Training did not restore at Epoch (%d) and step (%d)"
                        % (self.initial_epoch, self.initial_step)
                    )

        model = keras.Sequential([keras.layers.Dense(10)])
        optimizer = sgd.SGD()
        model.compile(optimizer, loss="mse")

        x = tf.random.uniform((24, 10))
        y = tf.random.uniform((24,))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(2)
        save_freq_arg = "epoch" if mode == "epoch" else 7
        backup_callback = cls(
            backup_dir=self.get_temp_dir(), save_freq=save_freq_arg
        )
        # epoch where the restore should resume from
        if save_freq_arg == "epoch":
            init_epoch = epoch_int
            init_step = 0
        elif save_freq_arg:
            init_epoch = int(((steps_int // 7) * 7) // 5)
            init_step = int((((steps_int // 7) * 7) % 5) - 1)
        else:
            init_epoch = 0
            init_step = 0

        # callback to verify accurate training state restore
        verify_restore_callback = VerifyRestore(
            initial_epoch=init_epoch, initial_step=init_step
        )
        try:
            model.fit(
                dataset,
                epochs=20,
                steps_per_epoch=5,
                callbacks=[backup_callback, InterruptingCallback()],
            )
        except RuntimeError as e:
            if str(e) == "EpochInterruption":
                logging.warning("***Handling interruption at epoch***")
            elif str(e) == "StepsInterruption":
                logging.warning("***Handling interruption at Nth step***")
            # This continues at the epoch and step where it left off.
            model.fit(
                dataset,
                epochs=20,
                steps_per_epoch=5,
                callbacks=[backup_callback, verify_restore_callback],
            )

    def test_experimental_backup_and_restore(self):
        """Ensure the legacy endpoint of `BackupAndRestore` gives warning."""

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_with(
                BackupAndRestoreExperimental
            )

        warning_msg = (
            "`tf.keras.callbacks.experimental.BackupAndRestore` "
            "endpoint is deprecated"
        )
        self.assertIn(warning_msg, "\n".join(warning_messages))
        warning_msg = "***Handling interruption***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

    def test_backup_and_restore(self):
        """Ensure the public endpoint of `BackupAndRestore` is working."""

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_with(BackupAndRestore)

        warning_msg = (
            "`tf.keras.callbacks.experimental.BackupAndRestore` "
            "endpoint is deprecated"
        )
        self.assertNotIn(warning_msg, "\n".join(warning_messages))
        warning_msg = "***Handling interruption***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

    def test_backup_and_restore_steps(self):
        """Ensure the public endpoint of `BackupAndRestore` is working."""

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            # interrupt at steps before 1 epoch
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=20, steps_int=3, mode="batch"
            )
        warning_msg = (
            "`tf.keras.callbacks.experimental.BackupAndRestore` "
            "endpoint is deprecated"
        )
        self.assertNotIn(warning_msg, "\n".join(warning_messages))
        warning_msg = "***Handling interruption at Nth step***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

        # interrupt at steps after 1 epoch
        warning_messages = []
        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=20, steps_int=8, mode="batch"
            )
        warning_msg = "***Handling interruption at Nth step***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

        # interrupt at epoch before steps
        warning_messages = []
        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=1, steps_int=12, mode="epoch"
            )
        warning_msg = "***Handling interruption at epoch***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

    def test_backup_and_restore_steps_last_batch(self):
        """Ensure the public endpoint of `BackupAndRestore` is working."""

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            # interrupt at last step in 7th epoch
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=20, steps_int=35, mode="batch"
            )
        warning_msg = (
            "`tf.keras.callbacks.experimental.BackupAndRestore` "
            "endpoint is deprecated"
        )
        self.assertNotIn(warning_msg, "\n".join(warning_messages))
        warning_msg = "***Handling interruption at Nth step***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

    def test_backup_and_restore_steps_false_save_freq(self):
        """Ensure the public endpoint of `BackupAndRestore` is working."""
        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            # interrupt at steps before 1 epoch
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=20, steps_int=3, mode=False
            )
        warning_msg = (
            "`tf.keras.callbacks.experimental.BackupAndRestore` "
            "endpoint is deprecated"
        )
        self.assertNotIn(warning_msg, "\n".join(warning_messages))
        warning_msg = "***Handling interruption at Nth step***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

        # interrupt at steps after 1 epoch
        warning_messages = []
        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=20, steps_int=8, mode="batch"
            )
        warning_msg = "***Handling interruption at Nth step***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

        # interrupt at epoch before steps
        warning_messages = []
        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            self._test_backup_and_restore_callback_at_steps(
                BackupAndRestore, epoch_int=1, steps_int=12, mode="epoch"
            )
        warning_msg = "***Handling interruption at epoch***"
        self.assertIn(warning_msg, "\n".join(warning_messages))

    def test_backup_and_restore_steps_clean_up(self):
        if not tf.executing_eagerly():
            self.skipTest(
                "BackupAndRestore only available when eager execution is "
                "enabled."
            )
        path = self.get_temp_dir()
        callback = BackupAndRestore(path, delete_checkpoint=True)
        model = keras.Sequential([keras.layers.Dense(10)])
        optimizer = gradient_descent.SGD()
        model.compile(optimizer, loss="mse")

        x = tf.random.uniform((24, 10))
        y = tf.random.uniform((24,))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        model.fit(dataset, epochs=1, callbacks=[callback])
        self.assertEmpty(os.listdir(path))

        callback = BackupAndRestore(path, delete_checkpoint=False)
        model.fit(dataset, epochs=1, callbacks=[callback])
        self.assertNotEmpty(os.listdir(path))

    @test_combinations.run_all_keras_modes
    def test_callback_warning(self):
        class SleepCallback(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                time.sleep(0.1)

        model = sequential.Sequential()
        model.add(keras.layers.Dense(1))
        model.compile(
            "sgd", loss="mse", run_eagerly=test_utils.should_run_eagerly()
        )

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            model.fit(
                np.ones((16, 1), "float32"),
                np.ones((16, 1), "float32"),
                batch_size=3,
                epochs=1,
                callbacks=[SleepCallback()],
            )
        warning_msg = (
            "Callback method `on_train_batch_end` is slow compared "
            "to the batch time"
        )
        self.assertIn(warning_msg, "\n".join(warning_messages))

    @test_combinations.run_all_keras_modes
    def test_default_callbacks_no_warning(self):
        # Test that without the callback no warning is raised
        model = sequential.Sequential()
        model.add(keras.layers.Dense(1))
        model.compile(
            "sgd", loss="mse", run_eagerly=test_utils.should_run_eagerly()
        )

        warning_messages = []

        def warning(msg):
            warning_messages.append(msg)

        with tf.compat.v1.test.mock.patch.object(logging, "warning", warning):
            model.fit(
                np.ones((16, 1), "float32"),
                np.ones((16, 1), "float32"),
                batch_size=3,
                epochs=1,
            )
        self.assertListEqual(warning_messages, [])

    @test_combinations.run_with_all_model_types(exclude_models="functional")
    @test_combinations.run_all_keras_modes
    def test_progbar_logging_deferred_model_build(self):
        model = self._get_model()
        self.assertFalse(model.built)

        x = tf.ones((200, 3))
        y = tf.zeros((200, 2))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        expected_log = r"(.*- loss:.*- my_acc:.*)+"

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(dataset, epochs=2, steps_per_epoch=10)
            self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    def test_progbar_logging_validation_data(self):
        model = self._get_model(input_shape=(3,))

        x = tf.ones((50, 3))
        y = tf.zeros((50, 2))
        training_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        val_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        expected_log = (
            r"(.*5/5.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*)+"
        )

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(training_dataset, epochs=2, validation_data=val_dataset)
            self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_progbar_logging_validation_split(self):
        model = self._get_model(input_shape=(3,))

        x = np.ones((100, 3))
        y = np.zeros((100, 2))
        expected_log = (
            r"(?s).*1/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:"
            r".*2/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*"
        )

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(x, y, batch_size=10, epochs=2, validation_split=0.2)
            self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_progbar_logging_training_validation(self):
        model = self._get_model(input_shape=(2,))

        def generator():
            for _ in range(100):
                yield [1, 1], 1

        training = (
            tf.data.Dataset.from_generator(
                generator=generator,
                output_types=("float64", "float64"),
                output_shapes=([2], []),
            )
            .batch(2)
            .repeat()
        )
        validation = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=("float64", "float64"),
            output_shapes=([2], []),
        ).batch(2)
        expected_log = (
            r"(?s).*1/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:"
            r".*2/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*"
        )

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(
                x=training,
                validation_data=validation,
                epochs=2,
                steps_per_epoch=20,
            )
            self.assertRegex(printed.contents(), expected_log)

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_progbar_logging_with_dataset_and_partial_batch(self):
        model = self._get_model(input_shape=(2,))

        def generator():
            # Have a partial batch at the end.
            for _ in range(9):
                yield np.random.random(2), 1

        training = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=("float64", "float64"),
            output_shapes=([2], []),
        ).batch(2)
        validation = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=("float64", "float64"),
            output_shapes=([2], []),
        ).batch(2)

        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(x=training, validation_data=validation)

            # Make sure the value of val_ metrics are not zeros.
            log_content = printed.contents()
            val_loss = re.findall(r"val_loss: (\d\.\d+)", log_content)
            self.assertLen(val_loss, 1)
            self.assertGreater(float(val_loss[0]), 0.0)

    @test_combinations.run_with_all_model_types
    def test_ModelCheckpoint(self):
        if h5py is None:
            return  # Skip test if models cannot be saved.

        model_type = test_utils.get_model_type()
        if model_type == "subclass":
            # Skip test since subclassed models cannot be saved in .h5 format.
            return
        if not tf.__internal__.tf2.enabled():
            self.skipTest("Checkpoint callback only available in v2.")

        layers = [
            keras.layers.Dense(
                NUM_HIDDEN, input_dim=INPUT_DIM, activation="relu"
            ),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(3,))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )

        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

        # Save model to a subdir inside the temp_dir so we can test
        # automatic directory creation.
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.h5")
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)

        # Case 1
        monitor = "val_loss"
        save_best_only = False
        mode = "auto"

        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 2
        mode = "min"
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 3
        mode = "max"
        monitor = "val_acc"
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 4
        save_best_only = True
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 5: metric not available.
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath, monitor="unknown", save_best_only=True
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        # File won't be written.
        assert not os.path.exists(filepath)

        # Case 6
        save_best_only = False
        period = 2
        mode = "auto"

        filepath = os.path.join(temp_dir, "checkpoint.{epoch:02d}.h5")
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                period=period,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=4,
            verbose=1,
        )
        assert os.path.exists(filepath.format(epoch=2))
        assert os.path.exists(filepath.format(epoch=4))
        os.remove(filepath.format(epoch=2))
        os.remove(filepath.format(epoch=4))
        assert not os.path.exists(filepath.format(epoch=1))
        assert not os.path.exists(filepath.format(epoch=3))

        # Invalid use: this will raise a warning but not an Exception.
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode="unknown",
        )

        # Case 7: `ModelCheckpoint` with a combination of `save_freq` and
        # `period`.  Though `period` is deprecated, we're testing it for
        # backward-compatibility.
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.h5")
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                mode=mode,
                save_freq="epoch",
                period=5,
            )
        ]
        assert not os.path.exists(filepath.format(epoch=0))
        assert not os.path.exists(filepath.format(epoch=5))
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=1,
        )
        assert not os.path.exists(filepath.format(epoch=1))
        assert not os.path.exists(filepath.format(epoch=2))
        assert not os.path.exists(filepath.format(epoch=3))
        assert not os.path.exists(filepath.format(epoch=4))
        assert os.path.exists(filepath.format(epoch=5))
        assert not os.path.exists(filepath.format(epoch=6))
        assert os.path.exists(filepath.format(epoch=10))
        os.remove(filepath.format(epoch=5))
        os.remove(filepath.format(epoch=10))

        # Case 8: `ModelCheckpoint` with an integer `save_freq`
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.h5")
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq=15,
                period=100,
            )  # The period should be ignored (this test tests this).
        ]
        assert not os.path.exists(filepath.format(epoch=3))
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=1,
        )
        assert not os.path.exists(filepath.format(epoch=1))
        assert not os.path.exists(filepath.format(epoch=2))
        assert os.path.exists(filepath.format(epoch=3))
        assert not os.path.exists(filepath.format(epoch=4))
        assert not os.path.exists(filepath.format(epoch=5))
        assert os.path.exists(filepath.format(epoch=6))
        assert not os.path.exists(filepath.format(epoch=7))
        assert not os.path.exists(filepath.format(epoch=8))
        assert os.path.exists(filepath.format(epoch=9))
        os.remove(filepath.format(epoch=3))
        os.remove(filepath.format(epoch=6))
        os.remove(filepath.format(epoch=9))

        # Case 9: `ModelCheckpoint` with valid and invalid save_freq argument.
        with self.assertRaisesRegex(ValueError, "Unrecognized save_freq"):
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq="invalid_save_freq",
            )
        # The following should not raise ValueError.
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq="epoch",
        )
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=3,
        )

        # Case 10: `ModelCheckpoint` with valid and invalid `options` argument.
        with self.assertRaisesRegex(TypeError, "tf.train.CheckpointOptions"):
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=True,
                mode=mode,
                options=tf.saved_model.SaveOptions(),
            )
        with self.assertRaisesRegex(TypeError, "tf.saved_model.SaveOptions"):
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=False,
                mode=mode,
                options=tf.train.CheckpointOptions(),
            )
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=True,
            mode=mode,
            options=tf.train.CheckpointOptions(),
        )
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode=mode,
            options=tf.saved_model.SaveOptions(),
        )

        # Case 11: `ModelCheckpoint` save model with batch number in filename.
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.h5"
        )
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=1
            )
        ]
        assert not os.path.exists(filepath.format(epoch=1, batch=1))
        assert not os.path.exists(filepath.format(epoch=1, batch=2))
        assert not os.path.exists(filepath.format(epoch=2, batch=1))
        assert not os.path.exists(filepath.format(epoch=2, batch=2))
        assert not os.path.exists(filepath.format(epoch=3, batch=1))
        assert not os.path.exists(filepath.format(epoch=3, batch=2))
        assert not os.path.exists(filepath.format(epoch=4, batch=1))
        assert not os.path.exists(filepath.format(epoch=4, batch=2))
        assert not os.path.exists(filepath.format(epoch=5, batch=1))
        assert not os.path.exists(filepath.format(epoch=5, batch=2))
        model.fit(
            x_train,
            y_train,
            batch_size=5,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=1,
        )

        assert os.path.exists(filepath.format(epoch=1, batch=1))
        assert os.path.exists(filepath.format(epoch=1, batch=2))
        assert os.path.exists(filepath.format(epoch=2, batch=1))
        assert os.path.exists(filepath.format(epoch=2, batch=2))
        assert os.path.exists(filepath.format(epoch=3, batch=1))
        assert os.path.exists(filepath.format(epoch=3, batch=2))
        assert os.path.exists(filepath.format(epoch=4, batch=1))
        assert os.path.exists(filepath.format(epoch=4, batch=2))
        assert os.path.exists(filepath.format(epoch=5, batch=1))
        assert os.path.exists(filepath.format(epoch=5, batch=2))

        os.remove(filepath.format(epoch=1, batch=1))
        os.remove(filepath.format(epoch=1, batch=2))
        os.remove(filepath.format(epoch=2, batch=1))
        os.remove(filepath.format(epoch=2, batch=2))
        os.remove(filepath.format(epoch=3, batch=1))
        os.remove(filepath.format(epoch=3, batch=2))
        os.remove(filepath.format(epoch=4, batch=1))
        os.remove(filepath.format(epoch=4, batch=2))
        os.remove(filepath.format(epoch=5, batch=1))
        os.remove(filepath.format(epoch=5, batch=2))

        # Case 12: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "max"
        monitor = "val_acc"
        initial_value_threshold = 0
        save_best_only = True
        filepath = os.path.join(temp_dir, "checkpoint.h5")
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 13: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = None
        save_best_only = True
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert os.path.exists(filepath)
        os.remove(filepath)

        # Case 14: ModelCheckpoint doesnt save model if loss was minimum earlier
        mode = "min"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert not os.path.exists(filepath)

        # Case 15: ModelCheckpoint doesnt save model if loss was min earlier in
        # auto mode
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            keras.callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        assert not os.path.exists(filepath)

    @test_utils.run_v2_only
    def test_ModelCheckpoint_subclass_SavedModel_save_weights_false(self):
        model = test_utils.get_small_subclass_mlp(NUM_HIDDEN, NUM_CLASSES)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        filepath = os.path.join(temp_dir, "checkpoint")
        cbks = [
            keras.callbacks.ModelCheckpoint(filepath, save_weights_only=False)
        ]

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_train = np_utils.to_categorical(y_train, num_classes=NUM_CLASSES)

        model.fit(x_train, y_train, callbacks=cbks, epochs=1, verbose=0)
        # Check that the filepath is a SavedModel directory.
        self.assertIn("saved_model.pb", os.listdir(filepath))

    @test_utils.run_v2_only
    def test_ModelCheckpoint_subclass_KerasV3_save_weights_false(self):
        model = test_utils.get_small_subclass_mlp(NUM_HIDDEN, NUM_CLASSES)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        filepath = os.path.join(temp_dir, "checkpoint.keras")
        cbks = [
            keras.callbacks.ModelCheckpoint(filepath, save_weights_only=False)
        ]

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_train = np_utils.to_categorical(y_train, num_classes=NUM_CLASSES)

        model.fit(x_train, y_train, callbacks=cbks, epochs=1, verbose=0)

        assert os.path.exists(filepath)

    def _get_dummy_resource_for_model_checkpoint_testing(self):
        def get_input_datasets():
            # Simple training input.
            train_input = [[1.0]] * 16
            train_label = [[0.0]] * 16
            ds = tf.data.Dataset.from_tensor_slices((train_input, train_label))
            return ds.batch(8, drop_remainder=True)

        # Very simple bias model to eliminate randomness.
        optimizer = gradient_descent.SGD(0.1)
        model = sequential.Sequential()
        model.add(test_utils.Bias(input_shape=(1,)))
        model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])
        train_ds = get_input_datasets()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.h5")

        # The filepath shouldn't exist at the beginning.
        self.assertFalse(os.path.exists(filepath))
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_weights_only=True
        )

        return model, train_ds, callback, filepath

    def _run_load_weights_on_restart_test_common_iterations(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()
        initial_epochs = 3
        model.fit(train_ds, epochs=initial_epochs, callbacks=[callback])

        # The files should exist after fitting with callback.
        for epoch in range(initial_epochs):
            self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))
        self.assertFalse(
            os.path.exists(filepath.format(epoch=initial_epochs + 1))
        )
        self.assertEqual(
            callback._get_most_recently_modified_file_matching_pattern(
                filepath
            ),
            filepath.format(epoch=initial_epochs),
        )

        model.fit(train_ds, epochs=1)
        weights_after_one_more_epoch = model.get_weights()

        # The filepath should continue to exist after fitting without callback.
        for epoch in range(initial_epochs):
            self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))

        return model, train_ds, filepath, weights_after_one_more_epoch

    @staticmethod
    def get_ModelCheckpoint_load_weights_on_restart_true_test(
        save_weights_only,
    ):
        def func(self):
            (
                model,
                train_ds,
                filepath,
                weights_after_one_more_epoch,
            ) = self._run_load_weights_on_restart_test_common_iterations()

            # Sleep for some short time period ensuring the files are created
            # with a different time (in MacOS OSS the granularity is only 1
            # second).
            time.sleep(2)
            callback = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=save_weights_only,
                load_weights_on_restart=True,
            )
            model.fit(train_ds, epochs=1, callbacks=[callback])
            weights_after_model_restoring_and_one_more_epoch = (
                model.get_weights()
            )

            self.assertEqual(
                callback._get_most_recently_modified_file_matching_pattern(
                    filepath
                ),
                filepath.format(epoch=1),
            )

            model.fit(
                train_ds,
                epochs=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=save_weights_only,
                        load_weights_on_restart=True,
                    )
                ],
            )
            weights_with_one_final_extra_epoch = model.get_weights()

            # Asserting the weights one epoch after initial fitting and another
            # epoch after that are closed, if a ModelCheckpoint with
            # load_weights_on_restart=True is given (so the model is restored at
            # the beginning of training).
            self.assertAllClose(
                weights_after_one_more_epoch,
                weights_after_model_restoring_and_one_more_epoch,
            )

            self.assertNotAllClose(
                weights_after_one_more_epoch, weights_with_one_final_extra_epoch
            )

        return func

    @staticmethod
    def get_ModelCheckpoint_load_weights_on_restart_false_test(
        save_weights_only,
    ):
        def func(self):
            (
                model,
                train_ds,
                filepath,
                weights_after_one_more_epoch,
            ) = self._run_load_weights_on_restart_test_common_iterations()

            model.fit(
                train_ds,
                epochs=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=filepath, save_weights_only=save_weights_only
                    )
                ],
            )
            weights_after_model_restoring_and_one_more_epoch = (
                model.get_weights()
            )

            # Asserting the weights one epoch after initial fitting and another
            # epoch after that are different, if a ModelCheckpoint with
            # load_weights_on_restart=False is given (so the model is not
            # restored at the beginning of training).
            self.assertNotAllClose(
                weights_after_one_more_epoch,
                weights_after_model_restoring_and_one_more_epoch,
            )

        return func

    test_model_checkpoint_load_weights_on_restart_true_save_weights_only_true = get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(  # noqa: E501
        True
    )

    test_model_checkpoint_load_weights_on_restart_true_save_weights_only_false = get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(  # noqa: E501
        False
    )

    test_model_checkpoint_load_weights_on_restart_false_save_weights_only_true = get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(  # noqa: E501
        True
    )

    test_model_checkpoint_load_weights_on_restart_false_save_weights_only_false = get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(  # noqa: E501
        False
    )

    def test_ModelCheckpoint_override_if_file_exist(self):
        (
            model,
            train_ds,
            filepath,
            _,
        ) = self._run_load_weights_on_restart_test_common_iterations()

        # Sleep for some short time period to ensure the files are created with
        # a different time (in MacOS OSS the granularity is only 1 second).
        time.sleep(2)
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_weights_only=True
        )
        model.load_weights(
            callback._get_most_recently_modified_file_matching_pattern(filepath)
        )
        weights_before_additional_fit = model.get_weights()
        model.fit(train_ds, epochs=1, callbacks=[callback])
        model.load_weights(
            callback._get_most_recently_modified_file_matching_pattern(filepath)
        )
        weights_after_additional_fit = model.get_weights()

        self.assertNotAllClose(
            weights_before_additional_fit, weights_after_additional_fit
        )

    def test_fit_with_ModelCheckpoint_with_tf_config(self):
        (
            model,
            train_ds,
            callback,
            _,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": {"worker": ["localhost:23333"]},
                "task": {"type": "worker", "index": 0},
            }
        )

        # `model.fit()` should work regardless of the presence of `TF_CONFIG`.
        model.fit(train_ds, epochs=1, callbacks=[callback])

    def test_fit_with_ModelCheckpoint_with_dir_as_h5_filepath(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "temp.h5")

        self.assertFalse(os.path.exists(filepath))
        os.mkdir(filepath)
        self.assertTrue(os.path.exists(filepath))

        callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

        with self.assertRaisesRegex(
            IOError,
            "Please specify a non-directory filepath for ModelCheckpoint.",
        ):
            model.fit(train_ds, epochs=1, callbacks=[callback])

    def test_ModelCheckpoint_KerasV3_save_options_error(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "temp.keras")

        with self.assertRaisesRegex(
            ValueError, "The native Keras format does not support"
        ):
            _ = keras.callbacks.ModelCheckpoint(
                filepath=filepath, options=tf.saved_model.SaveOptions()
            )

    def test_ModelCheckpoint_with_bad_path_placeholders(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "chkpt_{epoch:02d}_{mape:.2f}.h5")
        callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

        with self.assertRaisesRegex(
            KeyError, "Failed to format this callback filepath.*"
        ):
            model.fit(train_ds, epochs=1, callbacks=[callback])

    def test_ModelCheckpoint_nonblocking(self):
        filepath = self.get_temp_dir()
        # Should only cause a sync block when saving is actually performed.
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_freq=100
        )
        self.assertTrue(callback._supports_tf_logs)

        model = keras.Sequential([keras.layers.Dense(1)])
        cb_list = keras.callbacks.CallbackList(
            [callback], model=model, epochs=1, steps=10, verbose=0
        )

        tensor = tf.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, ModelCheckpoint is causing a blocking "
                "NumPy conversion even when not checkpointing."
            )

        tensor.numpy = mock_numpy

        logs = {"metric": tensor}

        cb_list.on_train_begin(logs)
        cb_list.on_epoch_begin(0, logs)
        cb_list.on_train_batch_begin(0, logs)
        cb_list.on_train_batch_end(0, logs)
        cb_list.on_epoch_end(0, logs)
        cb_list.on_train_end(logs)

        cb_list.on_test_begin(logs)
        cb_list.on_test_batch_begin(0, logs)
        cb_list.on_test_batch_end(0, logs)
        cb_list.on_test_end(logs)

        cb_list.on_predict_begin(logs)
        cb_list.on_predict_batch_begin(logs)
        cb_list.on_predict_batch_end(logs)
        cb_list.on_predict_end(logs)

    def test_verbose_2_logging(self):
        data = np.random.random((100, 1))
        labels = np.where(data > 0.5, 1, 0)
        model = keras.models.Sequential(
            (
                keras.layers.Dense(1, input_dim=1, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(
            optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]
        )
        expected_log = r"(.*- loss:.*- acc.*:.*epoch)+"
        with self.captureWritesToStream(sys.stdout) as printed:
            model.fit(data, labels, verbose=2, epochs=20)
            self.assertRegex(printed.contents(), expected_log)

    def test_ProgbarLogger_verbose_2_nonblocking(self):
        # Should only cause a sync block on epoch end methods.
        callback = keras.callbacks.ProgbarLogger(count_mode="steps")
        self.assertTrue(callback._supports_tf_logs)

        model = keras.Sequential([keras.layers.Dense(1)])
        cb_list = keras.callbacks.CallbackList(
            [callback], model=model, epochs=1, steps=10, verbose=2
        )

        tensor = tf.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, ModelCheckpoint is causing a blocking "
                "NumPy conversion even when not checkpointing."
            )

        tensor.numpy = mock_numpy
        logs = {"metric": tensor}

        cb_list.on_train_begin(logs)
        cb_list.on_epoch_begin(0, logs)
        cb_list.on_train_batch_begin(0, logs)
        cb_list.on_train_batch_end(0, logs)

        cb_list.on_test_begin(logs)
        cb_list.on_test_batch_begin(0, logs)
        cb_list.on_test_batch_end(0, logs)
        cb_list.on_test_end(logs)

        with self.assertRaisesRegex(RuntimeError, "NumPy conversion"):
            # on_epoch_end should still block.
            cb_list.on_epoch_end(0, logs)
        cb_list.on_train_end(logs)

    def test_EarlyStopping(self):
        with self.cached_session():
            np.random.seed(123)
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)
            model = test_utils.get_small_sequential_mlp(
                num_hidden=NUM_HIDDEN,
                num_classes=NUM_CLASSES,
                input_dim=INPUT_DIM,
            )
            model.compile(
                loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["acc"],
            )

            cases = [
                ("max", "val_acc"),
                ("min", "val_loss"),
                ("auto", "val_acc"),
                ("auto", "loss"),
                ("unknown", "unknown"),
            ]
            for mode, monitor in cases:
                patience = 0
                cbks = [
                    keras.callbacks.EarlyStopping(
                        patience=patience, monitor=monitor, mode=mode
                    )
                ]
                model.fit(
                    x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=cbks,
                    epochs=5,
                    verbose=0,
                )

    def test_EarlyStopping_patience(self):
        cases = [0, 1, 2, 3]
        losses = [10.0, 9.0, 8.0, 9.0, 8.9, 8.8, 8.7, 8.6, 8.5]

        for patience in cases:
            stopper = keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience
            )
            stopper.model = keras.models.Sequential()
            stopper.on_train_begin()

            for epoch, loss in enumerate(losses):
                stopper.on_epoch_end(epoch=epoch, logs={"loss": loss})
                if stopper.model.stop_training:
                    break

            self.assertEqual(stopper.stopped_epoch, max(patience, 1) + 2)

    def test_EarlyStopping_reuse(self):
        with self.cached_session():
            np.random.seed(1337)
            patience = 3
            data = np.random.random((100, 1))
            labels = np.where(data > 0.5, 1, 0)
            model = keras.models.Sequential(
                (
                    keras.layers.Dense(1, input_dim=1, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid"),
                )
            )
            model.compile(
                optimizer="sgd",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            weights = model.get_weights()

            # This should allow training to go for at least `patience` epochs
            model.set_weights(weights)

            stopper = keras.callbacks.EarlyStopping(
                monitor="acc", patience=patience
            )
            hist = model.fit(
                data, labels, callbacks=[stopper], verbose=0, epochs=20
            )
            assert len(hist.epoch) >= patience

    def test_EarlyStopping_with_baseline(self):
        with self.cached_session():
            np.random.seed(1337)
            baseline = 0.6
            (data, labels), _ = test_utils.get_test_data(
                train_samples=100,
                test_samples=50,
                input_shape=(1,),
                num_classes=NUM_CLASSES,
            )
            model = test_utils.get_small_sequential_mlp(
                num_hidden=1, num_classes=1, input_dim=1
            )
            model.compile(
                optimizer="sgd", loss="binary_crossentropy", metrics=["acc"]
            )

            stopper = keras.callbacks.EarlyStopping(
                monitor="acc", baseline=baseline
            )
            hist = model.fit(
                data, labels, callbacks=[stopper], verbose=0, epochs=20
            )
            assert len(hist.epoch) == 2

            patience = 3
            stopper = keras.callbacks.EarlyStopping(
                monitor="acc", patience=patience, baseline=baseline
            )
            hist = model.fit(
                data, labels, callbacks=[stopper], verbose=0, epochs=20
            )
            assert len(hist.epoch) >= patience

    def test_EarlyStopping_final_weights_when_restoring_model_weights(self):
        class DummyModel:
            def __init__(self):
                self.stop_training = False
                self.weights = -1

            def get_weights(self):
                return self.weights

            def set_weights(self, weights):
                self.weights = weights

            def set_weight_to_epoch(self, epoch):
                self.weights = epoch

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
        early_stop.model = DummyModel()
        losses = [0.2, 0.15, 0.1, 0.11, 0.12]
        # The best configuration is in the epoch 2 (loss = 0.1000).
        epochs_trained = 0
        early_stop.on_train_begin()
        for epoch in range(len(losses)):
            epochs_trained += 1
            early_stop.model.set_weight_to_epoch(epoch=epoch)
            early_stop.on_epoch_end(epoch, logs={"val_loss": losses[epoch]})
            if early_stop.model.stop_training:
                break
        # The best configuration is in epoch 2 (loss = 0.1000),
        # and while patience = 2, we're restoring the best weights,
        # so we end up at the epoch with the best weights, i.e. epoch 2
        self.assertEqual(early_stop.model.get_weights(), 2)

        # Check early stopping when no model beats the baseline.
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            baseline=0.5,
            restore_best_weights=True,
        )
        early_stop.model = DummyModel()
        losses = [0.9, 0.8, 0.7, 0.71, 0.72, 0.73]
        # The best configuration is in the epoch 2 (loss = 0.7000).
        epochs_trained = 0
        early_stop.on_train_begin()
        for epoch in range(len(losses)):
            epochs_trained += 1
            early_stop.model.set_weight_to_epoch(epoch=epoch)
            early_stop.on_epoch_end(epoch, logs={"val_loss": losses[epoch]})
            if early_stop.model.stop_training:
                break
        # No epoch improves on the baseline, so we should train for only 5
        # epochs, and restore the second model.
        self.assertEqual(epochs_trained, 5)
        self.assertEqual(early_stop.model.get_weights(), 2)

    def test_EarlyStopping_with_start_from_epoch(self):
        with self.cached_session():
            np.random.seed(1337)
            (data, labels), _ = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            labels = np_utils.to_categorical(labels)
            model = test_utils.get_small_sequential_mlp(
                num_hidden=NUM_HIDDEN,
                num_classes=NUM_CLASSES,
                input_dim=INPUT_DIM,
            )
            model.compile(
                optimizer="sgd", loss="binary_crossentropy", metrics=["acc"]
            )
            start_from_epoch = 2
            patience = 3
            stopper = keras.callbacks.EarlyStopping(
                monitor="acc",
                patience=patience,
                start_from_epoch=start_from_epoch,
            )
            history = model.fit(
                data, labels, callbacks=[stopper], verbose=0, epochs=20
            )
            # Test 'patience' argument functions correctly when used
            # in conjunction with 'start_from_epoch'.
            self.assertGreaterEqual(
                len(history.epoch), patience + start_from_epoch
            )

            start_from_epoch = 2
            patience = 0
            stopper = keras.callbacks.EarlyStopping(
                monitor="acc",
                patience=patience,
                start_from_epoch=start_from_epoch,
            )
            history = model.fit(
                data, labels, callbacks=[stopper], verbose=0, epochs=20
            )
            # Test for boundary condition when 'patience' = 0.
            self.assertGreaterEqual(len(history.epoch), start_from_epoch)

    def test_RemoteMonitor(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
            return None

        monitor = keras.callbacks.RemoteMonitor()
        # This will raise a warning since the default address in unreachable:
        monitor.on_epoch_end(0, logs={"loss": 0.0})

    def test_LearningRateScheduler(self):
        with self.cached_session():
            np.random.seed(1337)
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)
            model = test_utils.get_small_sequential_mlp(
                num_hidden=NUM_HIDDEN,
                num_classes=NUM_CLASSES,
                input_dim=INPUT_DIM,
            )
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"],
            )

            cbks = [
                keras.callbacks.LearningRateScheduler(
                    lambda x: 1.0 / (1.0 + x), verbose=1
                )
            ]
            io_utils.enable_interactive_logging()
            with self.captureWritesToStream(sys.stdout) as printed:
                model.fit(
                    x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=cbks,
                    epochs=5,
                )
                self.assertIn(
                    "LearningRateScheduler setting learning rate to 1.0",
                    printed.contents(),
                )
            assert (
                float(keras.backend.get_value(model.optimizer.lr)) - 0.2
            ) < keras.backend.epsilon()

            cbks = [keras.callbacks.LearningRateScheduler(lambda x, lr: lr / 2)]
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"],
            )
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0,
            )
            assert (
                float(keras.backend.get_value(model.optimizer.lr)) - 0.01 / 4
            ) < keras.backend.epsilon()

            cbks = [
                keras.callbacks.LearningRateScheduler(
                    lambda epoch, _: learning_rate_schedule.CosineDecay(
                        0.01, 2
                    )(epoch)
                )
            ]
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"],
            )
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0,
            )

            cosine_decay_np = 0.5 * (1 + np.cos(np.pi * (1 / 2)))
            decayed_learning_rate = 0.01 * cosine_decay_np

            assert (
                float(keras.backend.get_value(model.optimizer.lr))
                - decayed_learning_rate
            ) < keras.backend.epsilon()

    def test_ReduceLROnPlateau(self):
        with self.cached_session():
            tf_utils.set_random_seed(1337)
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)

            def make_model():
                tf_utils.set_random_seed(1337)
                model = test_utils.get_small_sequential_mlp(
                    num_hidden=NUM_HIDDEN,
                    num_classes=NUM_CLASSES,
                    input_dim=INPUT_DIM,
                )
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=gradient_descent.SGD(lr=0.1),
                )
                return model

            # TODO(psv): Make sure the callback works correctly when min_delta
            # is set as 0. Test fails when the order of this callback and
            # assertion is interchanged.
            model = make_model()
            cbks = [
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    min_delta=0,
                    patience=1,
                    cooldown=5,
                )
            ]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0,
            )
            self.assertAllClose(
                float(keras.backend.get_value(model.optimizer.lr)),
                0.1,
                atol=1e-4,
            )

            model = make_model()
            # This should reduce the LR after the first epoch (due to high
            # epsilon).
            cbks = [
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    min_delta=10,
                    patience=1,
                    cooldown=5,
                )
            ]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=2,
            )
            self.assertAllClose(
                float(keras.backend.get_value(model.optimizer.lr)),
                0.01,
                atol=1e-4,
            )

    def test_ReduceLROnPlateau_patience(self):
        class DummyOptimizer:
            def __init__(self):
                self.lr = keras.backend.variable(1.0)

        class DummyModel:
            def __init__(self):
                self.optimizer = DummyOptimizer()

        reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2
        )
        reduce_on_plateau.model = DummyModel()

        losses = [0.0860, 0.1096, 0.1040]
        lrs = []

        for epoch in range(len(losses)):
            reduce_on_plateau.on_epoch_end(
                epoch, logs={"val_loss": losses[epoch]}
            )
            lrs.append(
                keras.backend.get_value(reduce_on_plateau.model.optimizer.lr)
            )

        # The learning rates should be 1.0 except the last one
        for lr in lrs[:-1]:
            self.assertEqual(lr, 1.0)
        self.assertLess(lrs[-1], 1.0)

    def test_ReduceLROnPlateau_backwards_compatibility(self):
        with tf.compat.v1.test.mock.patch.object(
            logging, "warning"
        ) as mock_log:
            reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(epsilon=1e-13)
            self.assertRegex(
                str(mock_log.call_args), "`epsilon` argument is deprecated"
            )
        self.assertFalse(hasattr(reduce_on_plateau, "epsilon"))
        self.assertTrue(hasattr(reduce_on_plateau, "min_delta"))
        self.assertEqual(reduce_on_plateau.min_delta, 1e-13)

    def test_CSVLogger(self):
        with self.cached_session():
            np.random.seed(1337)
            temp_dir = self.get_temp_dir()
            self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
            filepath = os.path.join(temp_dir, "log.tsv")

            sep = "\t"
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)

            def make_model():
                np.random.seed(1337)
                model = test_utils.get_small_sequential_mlp(
                    num_hidden=NUM_HIDDEN,
                    num_classes=NUM_CLASSES,
                    input_dim=INPUT_DIM,
                )
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=gradient_descent.SGD(lr=0.1),
                    metrics=["accuracy"],
                )
                return model

            # case 1, create new file with defined separator
            model = make_model()
            cbks = [keras.callbacks.CSVLogger(filepath, separator=sep)]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=1,
                verbose=0,
            )

            assert os.path.exists(filepath)
            with open(filepath) as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read())
            assert dialect.delimiter == sep
            del model
            del cbks

            # case 2, append data to existing file, skip header
            model = make_model()
            cbks = [
                keras.callbacks.CSVLogger(filepath, separator=sep, append=True)
            ]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=1,
                verbose=0,
            )

            # case 3, reuse of CSVLogger object
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0,
            )

            with open(filepath) as csvfile:
                list_lines = csvfile.readlines()
                for line in list_lines:
                    assert line.count(sep) == 4
                assert len(list_lines) == 5
                output = " ".join(list_lines)
                assert len(re.findall("epoch", output)) == 1

            os.remove(filepath)

            # case 3, Verify Val. loss also registered when Validation Freq > 1
            model = make_model()
            cbks = [keras.callbacks.CSVLogger(filepath, separator=sep)]
            hist = model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                validation_freq=3,
                callbacks=cbks,
                epochs=5,
                verbose=0,
            )
            assert os.path.exists(filepath)
            # Verify that validation loss is registered at val. freq
            with open(filepath) as csvfile:
                rows = csv.DictReader(csvfile, delimiter=sep)
                for idx, row in enumerate(rows, 1):
                    self.assertIn("val_loss", row)
                    if idx == 3:
                        self.assertEqual(
                            row["val_loss"], str(hist.history["val_loss"][0])
                        )
                    else:
                        self.assertEqual(row["val_loss"], "NA")

    def test_stop_training_csv(self):
        # Test that using the CSVLogger callback with the TerminateOnNaN
        # callback does not result in invalid CSVs.
        np.random.seed(1337)
        tmpdir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        with self.cached_session():
            fp = os.path.join(tmpdir, "test.csv")
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )

            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)
            cbks = [
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.CSVLogger(fp),
            ]
            model = keras.models.Sequential()
            for _ in range(5):
                model.add(
                    keras.layers.Dense(
                        2, input_dim=INPUT_DIM, activation="relu"
                    )
                )
            model.add(keras.layers.Dense(NUM_CLASSES, activation="linear"))
            model.compile(loss="mean_squared_error", optimizer="rmsprop")

            def data_generator():
                i = 0
                max_batch_index = len(x_train) // BATCH_SIZE
                tot = 0
                while 1:
                    if tot > 3 * len(x_train):
                        yield (
                            np.ones([BATCH_SIZE, INPUT_DIM]) * np.nan,
                            np.ones([BATCH_SIZE, NUM_CLASSES]) * np.nan,
                        )
                    else:
                        yield (
                            x_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
                            y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
                        )
                    i += 1
                    tot += 1
                    i %= max_batch_index

            history = model.fit_generator(
                data_generator(),
                len(x_train) // BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=20,
            )
            loss = history.history["loss"]
            assert len(loss) > 1
            assert loss[-1] == np.inf or np.isnan(loss[-1])

            values = []
            with open(fp) as f:
                # On Windows, due to \r\n line ends, we may end up reading empty
                # lines after each line. Skip empty lines.
                values = [x for x in csv.reader(f) if x]

            assert "nan" in values[-1], "The last epoch was not logged."

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_TerminateOnNaN(self):
        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )

        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)
        cbks = [keras.callbacks.TerminateOnNaN()]
        model = keras.models.Sequential()
        initializer = keras.initializers.Constant(value=1e5)
        for _ in range(5):
            model.add(
                keras.layers.Dense(
                    2,
                    input_dim=INPUT_DIM,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        model.add(keras.layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="rmsprop")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))

    @unittest.skipIf(
        os.name == "nt",
        "use_multiprocessing=True does not work on windows properly.",
    )
    def test_LambdaCallback(self):
        with self.cached_session():
            np.random.seed(1337)
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)
            model = keras.models.Sequential()
            model.add(
                keras.layers.Dense(
                    NUM_HIDDEN, input_dim=INPUT_DIM, activation="relu"
                )
            )
            model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"],
            )

            # Start an arbitrary process that should run during model
            # training and be terminated after training has completed.
            e = threading.Event()

            def target():
                e.wait()

            t = threading.Thread(target=target)
            t.start()
            cleanup_callback = keras.callbacks.LambdaCallback(
                on_train_end=lambda logs: e.set()
            )

            cbks = [cleanup_callback]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=5,
                verbose=0,
            )
            t.join()
            assert not t.is_alive()

    def test_RemoteMonitor_np_array(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
        with tf.compat.v1.test.mock.patch.object(
            requests, "post"
        ) as requests_post:
            monitor = keras.callbacks.RemoteMonitor(send_as_json=True)
            a = np.arange(1)  # a 1 by 1 array
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitor_np_float32(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        with tf.compat.v1.test.mock.patch.object(
            requests, "post"
        ) as requests_post:
            monitor = keras.callbacks.RemoteMonitor(send_as_json=True)
            a = np.float32(1.0)  # a float32 generic type
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 1.0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitorWithJsonPayload(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
            return None
        with self.cached_session():
            (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES,
            )
            y_test = keras.utils.np_utils.to_categorical(y_test)
            y_train = keras.utils.np_utils.to_categorical(y_train)
            model = keras.models.Sequential()
            model.add(
                keras.layers.Dense(
                    NUM_HIDDEN, input_dim=INPUT_DIM, activation="relu"
                )
            )
            model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
            model.compile(
                loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"],
            )
            cbks = [keras.callbacks.RemoteMonitor(send_as_json=True)]

            with tf.compat.v1.test.mock.patch.object(requests, "post"):
                model.fit(
                    x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=cbks,
                    epochs=1,
                )

    def test_progbar_infers_steps(self):
        x, y = np.ones((10, 1)), np.ones((10, 1))
        data = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        data = data.filter(lambda x, y: True)  # Unknown cardinality.

        progbar = keras.callbacks.ProgbarLogger("steps")
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")
        self.assertIsNone(progbar.target)
        model.fit(data, epochs=2, callbacks=[progbar])
        self.assertEqual(progbar.target, 5)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_callback_passed_floats(self):
        class MyCallback(keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                assert isinstance(batch, int)
                assert isinstance(logs["loss"], float)
                self.on_batch_end_called = True

            def on_epoch_end(self, batch, logs=None):
                assert isinstance(batch, int)
                assert isinstance(logs["loss"], float)
                self.on_epoch_end_called = True

        x, y = np.ones((10, 1)), np.ones((10, 1))
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())

        callback = MyCallback()
        model.fit(x, y, epochs=2, callbacks=[callback])
        self.assertTrue(callback.on_batch_end_called)
        self.assertTrue(callback.on_batch_end_called)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_implements_batch_hooks(self):
        class MyCallbackWithBatchHooks(keras.callbacks.Callback):
            def __init__(self):
                self.train_batches = 0
                self.test_batches = 0
                self.predict_batches = 0

            def on_train_batch_end(self, batch, logs=None):
                self.train_batches += 1

            def on_test_batch_end(self, batch, logs=None):
                self.test_batches += 1

            def on_predict_batch_end(self, batch, logs=None):
                self.predict_batches += 1

        class MyCallbackWithTFBatchHooks(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self._supports_tf_logs = True

        class MyCallbackWithoutBatchHooks(keras.callbacks.Callback):
            def __init__(self):
                self.epochs = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epochs += 1

        x, y = np.ones((10, 1)), np.ones((10, 1))
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")

        my_cb = MyCallbackWithBatchHooks()
        cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
        self.assertTrue(cb_list._should_call_train_batch_hooks)
        self.assertTrue(cb_list._should_call_test_batch_hooks)
        self.assertTrue(cb_list._should_call_predict_batch_hooks)
        self.assertFalse(cb_list._batch_hooks_support_tf_logs)

        model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
        model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
        model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

        self.assertEqual(my_cb.train_batches, 2)
        self.assertEqual(my_cb.test_batches, 1)
        self.assertEqual(my_cb.predict_batches, 1)

        my_cb = MyCallbackWithTFBatchHooks()
        cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
        self.assertTrue(cb_list._batch_hooks_support_tf_logs)

        my_cb = MyCallbackWithoutBatchHooks()
        cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
        self.assertLen(cb_list.callbacks, 1)
        self.assertFalse(cb_list._should_call_train_batch_hooks)
        self.assertFalse(cb_list._should_call_test_batch_hooks)
        self.assertFalse(cb_list._should_call_predict_batch_hooks)

        model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
        model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
        model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_logs_conversion(self):
        assert_dict_equal = self.assertDictEqual

        class MutateNumpyLogs(CallAllHooks):
            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                logs["numpy"] = 1

        class MutateTensorFlowLogs(CallAllHooks):
            def __init__(self):
                super().__init__()
                self._supports_tf_logs = True

            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                logs["tf"] = 2

        class AssertNumpyLogs(CallAllHooks):
            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                assert_dict_equal(logs, {"all": 0, "numpy": 1, "tf": 2})

        class AssertTensorFlowLogs(AssertNumpyLogs):
            def __init__(self):
                super().__init__()
                self._supports_tf_logs = True

        cb_list = keras.callbacks.CallbackList(
            [
                MutateNumpyLogs(),
                MutateTensorFlowLogs(),
                AssertNumpyLogs(),
                AssertTensorFlowLogs(),
            ]
        )

        assert len(cb_list.callbacks) == 4
        cb_list.on_epoch_begin(0, logs={"all": 0})
        cb_list.on_epoch_end(0, logs={"all": 0})
        cb_list.on_predict_batch_begin(0, logs={"all": 0})
        cb_list.on_predict_batch_end(0, logs={"all": 0})
        cb_list.on_predict_begin(logs={"all": 0})
        cb_list.on_predict_end(logs={"all": 0})
        cb_list.on_test_batch_begin(0, logs={"all": 0})
        cb_list.on_test_batch_end(0, logs={"all": 0})
        cb_list.on_test_begin(logs={"all": 0})
        cb_list.on_test_end(logs={"all": 0})
        cb_list.on_train_batch_begin(0, logs={"all": 0})
        cb_list.on_train_batch_end(0, logs={"all": 0})
        cb_list.on_train_begin(logs={"all": 0})
        cb_list.on_train_end(logs={"all": 0})

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_implements_batch_hooks_override(self):
        class MyCallback(keras.callbacks.Callback):
            def __init__(self, should_run=True):
                self.should_run = should_run
                self.train_batches = 0
                self.test_batches = 0
                self.predict_batches = 0

            def on_train_batch_end(self, batch, logs=None):
                self.train_batches += 1

            def on_test_batch_end(self, batch, logs=None):
                self.test_batches += 1

            def on_predict_batch_end(self, batch, logs=None):
                self.predict_batches += 1

            def _implements_train_batch_hooks(self):
                return self.should_run

            def _implements_test_batch_hooks(self):
                return self.should_run

            def _implements_predict_batch_hooks(self):
                return self.should_run

        x, y = np.ones((10, 1)), np.ones((10, 1))
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")

        my_cb = MyCallback(should_run=True)
        cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
        self.assertTrue(cb_list._should_call_train_batch_hooks)
        self.assertTrue(cb_list._should_call_test_batch_hooks)
        self.assertTrue(cb_list._should_call_predict_batch_hooks)

        model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
        model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
        model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

        self.assertEqual(my_cb.train_batches, 2)
        self.assertEqual(my_cb.test_batches, 1)
        self.assertEqual(my_cb.predict_batches, 1)

        my_cb = MyCallback(should_run=False)
        cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
        self.assertFalse(cb_list._should_call_train_batch_hooks)
        self.assertFalse(cb_list._should_call_test_batch_hooks)
        self.assertFalse(cb_list._should_call_predict_batch_hooks)

        model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
        model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
        model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

        self.assertEqual(my_cb.train_batches, 0)
        self.assertEqual(my_cb.test_batches, 0)
        self.assertEqual(my_cb.predict_batches, 0)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_default_callbacks_do_not_call_batch_hooks(self):
        model = keras.Sequential([keras.layers.Dense(1)])
        log_dir = self.get_temp_dir()
        cb_list = keras.callbacks.CallbackList(
            [
                keras.callbacks.TensorBoard(log_dir, profile_batch=0),
                keras.callbacks.ModelCheckpoint(log_dir),
            ],
            add_progbar=True,
            model=model,
            verbose=2,
            epochs=3,
        )
        self.assertLen(cb_list.callbacks, 3)
        self.assertFalse(cb_list._should_call_train_batch_hooks)
        self.assertFalse(cb_list._should_call_test_batch_hooks)
        self.assertFalse(cb_list._should_call_predict_batch_hooks)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_change_tf_functions_during_fit(self):
        class ChangeFunctions(keras.callbacks.Callback):
            def on_epoch_end(self, epochs, logs=None):
                def new_fn(iterator):
                    raise ValueError("New function substituted successfully.")

                self.model.train_function = new_fn
                self.model.test_function = new_fn
                self.model.predict_function = new_fn

        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")

        x, y = np.ones((10, 10)), np.ones((10, 1))
        with self.assertRaisesRegex(ValueError, "New function "):
            model.fit(
                x, y, batch_size=2, epochs=2, callbacks=[ChangeFunctions()]
            )
        with self.assertRaisesRegex(ValueError, "New function "):
            model.evaluate(x, y, batch_size=2)
        with self.assertRaisesRegex(ValueError, "New function "):
            model.predict(x, batch_size=2)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_stop_training_batch_level(self):
        class MyCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.batch_counter = 0

            def on_train_batch_end(self, batch, logs=None):
                self.batch_counter += 1
                if batch == 2:
                    self.model.stop_training = True

        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile("sgd", "mse")
        x, y = np.ones((10, 10)), np.ones((10, 1))
        my_cb = MyCallback()
        # Will run 5 batches if `stop_training` doesn't work.
        model.fit(x, y, batch_size=2, callbacks=[my_cb])
        self.assertEqual(my_cb.batch_counter, 3)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_built_in_callback_order(self):
        class CustomCallback(keras.callbacks.Callback):
            pass

        class TestingCallbackList(keras.callbacks.CallbackList):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if (
                    (not isinstance(self.callbacks[0], CustomCallback))
                    or (
                        not isinstance(
                            self.callbacks[1], keras.callbacks.History
                        )
                    )
                    or (
                        not isinstance(
                            self.callbacks[2], keras.callbacks.ProgbarLogger
                        )
                    )
                ):
                    raise AssertionError(
                        f"Callback order unexpected: {self.callbacks}"
                    )

        with mock.patch.object(
            keras.callbacks, "CallbackList", TestingCallbackList
        ):
            model = keras.Sequential([keras.layers.Dense(1)])
            model.compile("sgd", "mse")
            custom_callback = CustomCallback()
            model.fit(
                np.ones((10, 10)),
                np.ones((10, 1)),
                epochs=5,
                callbacks=[custom_callback],
            )


# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple("_ObservedSummary", ("logdir", "tag"))


class _SummaryFile:
    """A record of summary tags and the files to which they were written.

    Fields `scalars`, `images`, `histograms`, and `tensors` are sets
    containing `_ObservedSummary` values.
    """

    def __init__(self):
        self.scalars = set()
        self.images = set()
        self.histograms = set()
        self.tensors = set()
        self.graph_defs = []
        self.convert_from_v2_summary_proto = False


def list_summaries(logdir):
    """Read all summaries under the logdir into a `_SummaryFile`.

    Args:
      logdir: A path to a directory that contains zero or more event
        files, either as direct children or in transitive subdirectories.
        Summaries in these events must only contain old-style scalars,
        images, and histograms. Non-summary events, like `graph_def`s, are
        ignored.

    Returns:
      A `_SummaryFile` object reflecting all summaries written to any
      event files in the logdir or any of its descendant directories.

    Raises:
      ValueError: If an event file contains an summary of unexpected kind.
    """
    result = _SummaryFile()
    for dirpath, _, filenames in os.walk(logdir):
        for filename in filenames:
            if not filename.startswith("events.out."):
                continue
            path = os.path.join(dirpath, filename)
            for event in tf.compat.v1.train.summary_iterator(path):
                if event.graph_def:
                    result.graph_defs.append(event.graph_def)
                if not event.summary:  # (e.g., it's a `graph_def` event)
                    continue
                for value in event.summary.value:
                    tag = value.tag
                    # Case on the `value` rather than the summary metadata
                    # because the Keras callback uses `summary_ops_v2` to emit
                    # old-style summaries. See b/124535134.
                    kind = value.WhichOneof("value")
                    container = {
                        "simple_value": result.scalars,
                        "image": result.images,
                        "histo": result.histograms,
                        "tensor": result.tensors,
                    }.get(kind)
                    if container is None:
                        raise ValueError(
                            "Unexpected summary kind %r in event file %s:\n%r"
                            % (kind, path, event)
                        )
                    elif kind == "tensor" and tag != "keras":
                        # Convert the tf2 summary proto to old style for type
                        # checking.
                        plugin_name = value.metadata.plugin_data.plugin_name
                        container = {
                            "images": result.images,
                            "histograms": result.histograms,
                            "scalars": result.scalars,
                        }.get(plugin_name)
                        if container is not None:
                            result.convert_from_v2_summary_proto = True
                        else:
                            container = result.tensors
                    container.add(_ObservedSummary(logdir=dirpath, tag=tag))
    return result


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2(test_combinations.TestCase):
    def setUp(self):
        super(TestTensorBoardV2, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_model(self, compile_model=True):
        layers = [
            keras.layers.Conv2D(8, (3, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ]
        model = test_utils.get_model_from_layers(
            layers, input_shape=(10, 10, 1)
        )
        if compile_model:
            opt = gradient_descent.SGD(learning_rate=0.001)
            model.compile(
                opt, "mse", run_eagerly=test_utils.should_run_eagerly()
            )
        return model

    def test_TensorBoard_default_logdir(self):
        """Regression test for cross-platform pathsep in default logdir."""
        os.chdir(self.get_temp_dir())

        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard()  # no logdir specified

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(logdir=".")
        train_dir = os.path.join(".", "logs", "train")
        validation_dir = os.path.join(".", "logs", "validation")
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=validation_dir, tag="evaluation_loss_vs_iterations"
                ),
            },
        )

    def test_TensorBoard_basic(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_across_invocations(self):
        """Regression test for summary writer resource use-after-free.

        See: <https://github.com/tensorflow/tensorflow/issues/25707>
        """
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir)

        for _ in (1, 2):
            model.fit(
                x,
                y,
                batch_size=2,
                epochs=2,
                validation_data=(x, y),
                callbacks=[tb_cbk],
            )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_no_spurious_event_files(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir)

        model.fit(x, y, batch_size=2, epochs=2, callbacks=[tb_cbk])

        events_file_run_basenames = set()
        for dirpath, _, filenames in os.walk(self.train_dir):
            if any(fn.startswith("events.out.") for fn in filenames):
                events_file_run_basenames.add(os.path.basename(dirpath))
        self.assertEqual(events_file_run_basenames, {"train"})

    def test_TensorBoard_batch_metrics(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_learning_rate_schedules(self):
        model = self._get_model(compile_model=False)
        opt = gradient_descent.SGD(learning_rate_schedule.CosineDecay(0.01, 1))
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            callbacks=[keras.callbacks.TensorBoard(self.logdir)],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
            },
        )

    def test_TensorBoard_global_step(self):
        model = self._get_model(compile_model=False)
        opt = gradient_descent.SGD(learning_rate_schedule.CosineDecay(0.01, 1))
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            verbose=0,
            callbacks=[
                keras.callbacks.TensorBoard(
                    self.logdir,
                    update_freq=1,
                    profile_batch=0,
                    write_steps_per_second=True,
                )
            ],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_steps_per_second"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="batch_steps_per_second"
                ),
            },
        )

    def test_TensorBoard_weight_histograms(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)
        model_type = test_utils.get_model_type()

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/histogram"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="kernel_0/histogram"
                ),
            },
        )

    def test_TensorBoard_weight_images(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, write_images=True
        )
        model_type = test_utils.get_model_type()

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/histogram"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="kernel_0/histogram"
                ),
            },
        )
        if summary_file.convert_from_v2_summary_proto:
            expected_image_summaries = {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/image"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image"),
            }
        else:
            expected_image_summaries = {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/image/0"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/0"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/1"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/2"),
            }
        self.assertEqual(
            self._strip_layer_names(summary_file.images, model_type),
            expected_image_summaries,
        )

    def test_TensorBoard_projector_callback(self):
        layers = [
            keras.layers.Embedding(10, 10, name="test_embedding"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(10,))
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            embeddings_freq=1,
            embeddings_metadata={"test_embedding": "metadata.tsv"},
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        with open(os.path.join(self.logdir, "projector_config.pbtxt")) as f:
            self.assertEqual(
                f.readlines(),
                [
                    "embeddings {\n",
                    "  tensor_name: "
                    '"layer_with_weights-0/embeddings/.ATTRIBUTES/'
                    'VARIABLE_VALUE"\n',
                    '  metadata_path: "metadata.tsv"\n',
                    "}\n",
                ],
            )

    def test_custom_summary(self):
        if not tf.executing_eagerly():
            self.skipTest("Custom summaries only supported in V2 code path.")

        def scalar_v2_mock(name, data, step=None):
            """A reimplementation of the scalar plugin to avoid circular
            deps."""
            metadata = tf.compat.v1.SummaryMetadata()
            # Should match value in tensorboard/plugins/scalar/metadata.py.
            metadata.plugin_data.plugin_name = "scalars"
            with tf.summary.experimental.summary_scope(
                name, "scalar_summary", values=[data, step]
            ) as (tag, _):
                return tf.summary.write(
                    tag=tag,
                    tensor=tf.cast(data, "float32"),
                    step=step,
                    metadata=metadata,
                )

        class LayerWithSummary(keras.layers.Layer):
            def call(self, x):
                scalar_v2_mock("custom_summary", tf.reduce_sum(x))
                return x

        model = test_utils.get_model_from_layers(
            [LayerWithSummary()], input_shape=(5,), name="model"
        )

        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)
        x, y = np.ones((10, 5)), np.ones((10, 5))
        model.fit(
            x, y, batch_size=2, validation_data=(x, y), callbacks=[tb_cbk]
        )
        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
                _ObservedSummary(
                    logdir=self.train_dir,
                    tag="model/layer_with_summary/custom_summary",
                ),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="model/layer_with_summary/custom_summary",
                ),
            },
        )

    def _strip_layer_names(self, summaries, model_type):
        """Deduplicate summary names modulo layer prefix.

        This removes the first slash-component of each tag name: for
        instance, "foo/bar/baz" becomes "bar/baz".

        Args:
          summaries: A `set` of `_ObservedSummary` values.
          model_type: The model type currently being tested.

        Returns:
          A new `set` of `_ObservedSummary` values with layer prefixes
          removed.
        """
        result = set()
        for summary in summaries:
            if "/" not in summary.tag:
                raise ValueError(f"tag has no layer name: {summary.tag!r}")
            start_from = 2 if "subclass" in model_type else 1
            new_tag = "/".join(summary.tag.split("/")[start_from:])
            result.add(summary._replace(tag=new_tag))
        return result

    def test_TensorBoard_invalid_argument(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized arguments"):
            keras.callbacks.TensorBoard(wwrite_images=True)

    def test_TensorBoard_non_blocking(self):
        model = keras.Sequential([keras.layers.Dense(1)])
        tb = keras.callbacks.TensorBoard(self.logdir)
        self.assertTrue(tb._supports_tf_logs)
        cb_list = keras.callbacks.CallbackList(
            [tb], model=model, epochs=1, steps=100, verbose=0
        )

        tensor = tf.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, TensorBoard is causing a blocking "
                "NumPy conversion."
            )

        with tf.compat.v1.test.mock.patch.object(tensor, "numpy", mock_numpy):
            logs = {"metric": tensor}

            cb_list.on_train_begin(logs)
            cb_list.on_epoch_begin(0, logs)
            cb_list.on_train_batch_begin(0, logs)
            cb_list.on_train_batch_end(0, logs)
            cb_list.on_epoch_end(0, logs)
            cb_list.on_train_end(logs)

            cb_list.on_test_begin(logs)
            cb_list.on_test_batch_begin(0, logs)
            cb_list.on_test_batch_end(0, logs)
            cb_list.on_test_end(logs)

            cb_list.on_predict_begin(logs)
            cb_list.on_predict_batch_begin(logs)
            cb_list.on_predict_batch_end(logs)
            cb_list.on_predict_end(logs)


# Note that this test specifies model_type explicitly.
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2NonParameterizedTest(test_combinations.TestCase):
    def setUp(self):
        super(TestTensorBoardV2NonParameterizedTest, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_seq_model(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        opt = gradient_descent.SGD(learning_rate=0.001)
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())
        return model

    def _count_xplane_file(self, logdir):
        profile_dir = os.path.join(logdir, "plugins", "profile")
        count = 0
        for dirpath, dirnames, filenames in os.walk(profile_dir):
            del dirpath  # unused
            del dirnames  # unused
            for filename in filenames:
                if filename.endswith(".xplane.pb"):
                    count += 1
        return count

    def fitModelAndAssertKerasModelWritten(self, model):
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, write_graph=True, profile_batch=0
        )
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=3,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="keras"),
            },
        )
        if not model.run_eagerly:
            # There should be one train graph
            self.assertLen(summary_file.graph_defs, 1)
            for graph_def in summary_file.graph_defs:
                graph_def_str = str(graph_def)

                # All the model layers should appear in the graphs
                for layer in model.layers:
                    if "input" not in layer.name:
                        self.assertIn(layer.name, graph_def_str)

    def test_TensorBoard_writeSequentialModel_noInputShape(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_writeSequentialModel_withInputShape(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_writeModel(self):
        inputs = keras.layers.Input([10, 10, 1])
        x = keras.layers.Conv2D(8, (3, 3), activation="relu")(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)(x)
        model = keras.models.Model(inputs=inputs, outputs=[x])
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_autoTrace(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=1, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_1"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_outerProfiler(self):
        """Runs a profiler session that interferes with the callback's one.

        The callback will not generate a profile but execution will proceed
        without crashing due to unhandled exceptions.
        """
        tf.profiler.experimental.start(logdir="")
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=1, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)
        tf.profiler.experimental.stop(save=False)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_1"),
            },
        )
        self.assertEqual(0, self._count_xplane_file(logdir=self.train_dir))

    def test_TensorBoard_autoTrace_tagNameWithBatchNum(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=2, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_2"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileBatchRangeSingle(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="2,2",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                # Trace will be logged once at the batch it stops profiling.
                _ObservedSummary(logdir=self.train_dir, tag="batch_2"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileBatchRangeTwice(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="10,10",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=10,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        time.sleep(1)  # Avoids the second profile over-writing the first.

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=10,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        self.assertEqual(2, self._count_xplane_file(logdir=self.logdir))

    # Test case that replicates a GitHub issue.
    # https://github.com/tensorflow/tensorflow/issues/37543
    def test_TensorBoard_autoTrace_profileTwiceGraphMode(self):
        tf.compat.v1.disable_eager_execution()
        inp = keras.Input((1,))
        out = keras.layers.Dense(units=1)(inp)
        model = keras.Model(inp, out)

        model.compile(gradient_descent.SGD(1), "mse")

        logdir = os.path.join(self.get_temp_dir(), "tb1")
        model.fit(
            np.zeros((64, 1)),
            np.zeros((64, 1)),
            batch_size=32,
            callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=1)],
        )
        # Verifies trace exists in the first logdir.
        self.assertEqual(1, self._count_xplane_file(logdir=logdir))
        logdir = os.path.join(self.get_temp_dir(), "tb2")
        model.fit(
            np.zeros((64, 1)),
            np.zeros((64, 1)),
            batch_size=32,
            callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=2)],
        )
        # Verifies trace exists in the second logdir.
        self.assertEqual(1, self._count_xplane_file(logdir=logdir))

    def test_TensorBoard_autoTrace_profileBatchRange(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="1,3",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=4,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                # Trace will be logged once at the batch it stops profiling.
                _ObservedSummary(logdir=self.train_dir, tag="batch_3"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileInvalidBatchRange(self):
        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="-1,3",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="1,None",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="6,5",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch=-1,
                write_graph=False,
            )

    def test_TensorBoard_autoTrace_profile_batch_largerThanBatchCount(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch=10000,
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        # Enabled trace only on the 10000th batch, thus it should be empty.
        self.assertEmpty(summary_file.tensors)
        self.assertEqual(0, self._count_xplane_file(logdir=self.train_dir))


class MostRecentlyModifiedFileMatchingPatternTest(tf.test.TestCase):
    def test_get_most_recently_modified_file_matching_pattern(self):
        file_pattern = "f.batch{batch:02d}epoch{epoch:02d}.h5"
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name)
            for file_name in [
                "f.batch03epoch02.h5",
                "f.batch02epoch02.h5",
                "f.batch01epoch01.h5",
            ]
        ]
        for file_path in file_paths:
            with open(file_path, "w") as f:
                # Ensure there are some intervals between file creation.
                time.sleep(2)
                f.write("foo bar")
        # Ensure the files have been actually written.
        self.assertEqual(
            set(
                [
                    os.path.join(test_dir, file_name)
                    for file_name in os.listdir(test_dir)
                ]
            ),
            set(file_paths),
        )
        self.assertEqual(
            keras.callbacks.ModelCheckpoint(
                None
            )._get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1],
        )

    def test_some_file_not_matching_pattern(self):
        file_pattern = "f.batch{batch:02d}epoch{epoch:02d}.h5"
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name)
            for file_name in [
                "f.batch03epoch02.h5",
                "f.batch02epoch02.h5",
                "f.baatch01epoch01.h5",
            ]
        ]
        for file_path in file_paths:
            with open(file_path, "w") as f:
                # Ensure there are some intervals between file creation.
                time.sleep(2)
                f.write("foo bar")
        self.assertEqual(
            keras.callbacks.ModelCheckpoint(
                None
            )._get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-2],
        )

    def test_get_same_file_if_file_name_equals_pattern(self):
        file_name = "f.batch02.h5"
        test_dir = self.get_temp_dir()
        file_path = os.path.join(test_dir, file_name)
        with open(file_path, "w") as f:
            f.write("foo bar")
        self.assertEqual(
            os.path.join(test_dir, os.listdir(test_dir)[0]), file_path
        )
        self.assertEqual(
            keras.callbacks.ModelCheckpoint(
                None
            )._get_most_recently_modified_file_matching_pattern(file_path),
            file_path,
        )

    def test_get_none_if_file_does_not_exist(self):
        file_name = "f.batch02.h5"
        test_dir = self.get_temp_dir()
        file_path = os.path.join(test_dir, file_name)
        self.assertEmpty(os.listdir(test_dir))
        self.assertEqual(
            keras.callbacks.ModelCheckpoint(
                None
            )._get_most_recently_modified_file_matching_pattern(file_path),
            None,
        )

    def test_using_checkpoint_management_latest_checkpoint(self):
        file_pattern = "f.batch{batch:02d}epoch{epoch:02d}"
        ckpt_file_name = "f.batchXepochY"
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        ckpt_file_path = os.path.join(test_dir, ckpt_file_name)
        with open(ckpt_file_path, "w") as f:
            f.write("dummy ckpt")
        tf.__internal__.train.update_checkpoint_state(test_dir, ckpt_file_path)

        file_paths = [
            os.path.join(test_dir, file_name)
            for file_name in ["f.batch03epoch02", "f.batch02epoch02"]
        ]
        for file_path in file_paths:
            with open(file_path, "w") as f:
                f.write("foo bar")

        # The result returned from checkpoint_management.latest_checkpoint takes
        # priority, so even if it was written earlier, we should still return
        # that.
        self.assertEqual(
            keras.callbacks.ModelCheckpoint(
                None
            )._get_most_recently_modified_file_matching_pattern(path_pattern),
            ckpt_file_path,
        )


class SummaryOpsTest(tf.test.TestCase):
    def tearDown(self):
        super(SummaryOpsTest, self).tearDown()
        tf.summary.trace_off()

    def keras_model(self, *args, **kwargs):
        logdir = self.get_temp_dir()
        writer = tf.summary.create_file_writer(logdir)
        with writer.as_default():
            keras.callbacks.keras_model_summary(*args, **kwargs)
        writer.close()
        events = events_from_logdir(logdir)
        # The first event contains no summary values. The written content goes
        # to the second event.
        return events[1]

    @test_utils.run_v2_only
    def testKerasModel(self):
        model = keras.Sequential(
            [Dense(10, input_shape=(100,)), Activation("relu", name="my_relu")]
        )
        event = self.keras_model(name="my_name", data=model, step=1)
        first_val = event.summary.value[0]
        self.assertEqual(
            model.to_json(), first_val.tensor.string_val[0].decode()
        )

    @test_utils.run_v2_only
    def testKerasModel_usesDefaultStep(self):
        model = keras.Sequential(
            [Dense(10, input_shape=(100,)), Activation("relu", name="my_relu")]
        )
        try:
            tf.summary.experimental.set_step(42)
            event = self.keras_model(name="my_name", data=model)
            self.assertEqual(42, event.step)
        finally:
            # Reset to default state for other tests.
            tf.summary.experimental.set_step(None)

    @test_utils.run_v2_only
    def testKerasModel_subclass(self):
        class SimpleSubclass(keras.Model):
            def __init__(self):
                super().__init__(name="subclass")
                self.dense = Dense(10, input_shape=(100,))
                self.activation = Activation("relu", name="my_relu")

            def call(self, inputs):
                x = self.dense(inputs)
                return self.activation(x)

            # Intentionally erroring out at json serialization to test the
            # warning.
            def get_config(self):
                raise NotImplementedError

        model = SimpleSubclass()
        with tf.compat.v1.test.mock.patch.object(
            logging, "warning"
        ) as mock_log:
            self.assertFalse(
                keras.callbacks.keras_model_summary(
                    name="my_name", data=model, step=1
                )
            )
            self.assertRegex(
                str(mock_log.call_args), "Model failed to serialize as JSON."
            )

    @test_utils.run_v2_only
    def testKerasModel_otherExceptions(self):
        model = keras.Sequential()

        with tf.compat.v1.test.mock.patch.object(
            model, "to_json"
        ) as mock_to_json:
            with tf.compat.v1.test.mock.patch.object(
                logging, "warning"
            ) as mock_log:
                mock_to_json.side_effect = Exception("oops")
                self.assertFalse(
                    keras.callbacks.keras_model_summary(
                        name="my_name", data=model, step=1
                    )
                )
                self.assertRegex(
                    str(mock_log.call_args),
                    "Model failed to serialize as JSON. Ignoring",
                )


def events_from_file(filepath):
    """Returns all events in a single event file.

    Args:
      filepath: Path to the event file.

    Returns:
      A list of all tf.Event protos in the event file.
    """
    result = []
    raw_dataset = tf.data.TFRecordDataset([filepath])
    for raw_record in raw_dataset.take(10):
        event = tf.compat.v1.Event()
        event.ParseFromString(raw_record.numpy())
        result.append(event)
    return result


def events_from_logdir(logdir):
    """Returns all events in the single eventfile in logdir.

    Args:
      logdir: The directory in which the single event file is sought.

    Returns:
      A list of all tf.Event protos from the single event file.

    Raises:
      AssertionError: If logdir does not contain exactly one file.
    """
    assert tf.compat.v1.gfile.Exists(logdir)
    files = tf.compat.v1.gfile.ListDirectory(logdir)
    assert len(files) == 1, f"Found not exactly one file in logdir: {files}"
    return events_from_file(os.path.join(logdir, files[0]))


if __name__ == "__main__":
    tf.test.main()
