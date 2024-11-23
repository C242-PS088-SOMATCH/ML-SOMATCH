"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import configuration
import bi_lstm as polyvore_model

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--input_file_pattern", type=str, required=True, help="File pattern of sharded TFRecord input files.")
parser.add_argument("--inception_checkpoint_file", type=str, default="", help="Path to a pretrained inception_v3 model.")
parser.add_argument("--train_dir", type=str, required=True, help="Directory for saving and loading model checkpoints.")
parser.add_argument("--train_inception", action="store_true", help="Whether to train inception submodel variables.")
parser.add_argument("--number_of_steps", type=int, default=1000000, help="Number of training steps.")
parser.add_argument("--log_every_n_steps", type=int, default=1, help="Frequency at which loss and global step are logged.")
args = parser.parse_args()

# Logging
tf.get_logger().setLevel('INFO')


def main():
    # Validate inputs
    assert args.input_file_pattern, "--input_file_pattern is required"
    assert args.train_dir, "--train_dir is required"

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = args.input_file_pattern
    model_config.inception_checkpoint_file = args.inception_checkpoint_file

    training_config = configuration.TrainingConfig()

    # Create training directory if it does not exist.
    train_dir = args.train_dir
    if not os.path.exists(train_dir):
        tf.get_logger().info("Creating training directory: %s", train_dir)
        os.makedirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = polyvore_model.PolyvoreModel(
            model_config, mode="train", train_inception=args.train_inception)
        model.build()
        
        # Configure learning rate and decay.
        learning_rate = tf.Variable(training_config.initial_learning_rate, trainable=False, dtype=tf.float32)
    
    
        learning_rate_decay_fn = None
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                    model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                                training_config.num_epochs_per_decay)

            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=training_config.initial_learning_rate,
                        decay_steps=decay_steps,
                        decay_rate=training_config.learning_rate_decay_factor,
                        staircase=True)
        else:
            learning_rate_schedule = training_config.initial_learning_rate

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        # Set up the training ops.
        train_op = optimizer.minimize(model.total_loss, var_list=model.trainable_variables)

        # Saver for saving and restoring model checkpoints
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, train_dir, max_to_keep=training_config.max_checkpoints_to_keep)

    # Training loop
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        if args.inception_checkpoint_file:
            model.init_fn(sess)

    for step in range(args.number_of_steps):
        _, loss = sess.run([train_op, model.total_loss])
        if step % args.log_every_n_steps == 0:
            tf.get_logger().info("Step %d: Loss = %.4f", step, loss)

            if step % 1000 == 0:  # Save checkpoint every 1000 steps
                checkpoint_manager.save()


if __name__ == "__main__":
    main()
