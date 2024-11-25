"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def parse_sequence_example(serialized, set_id, image_feature, image_index, caption_feature, number_set_images):
    """Parses a tensorflow.SequenceExample into a set of images and caption."""

    context_features = {
        set_id: tf.io.FixedLenFeature([], dtype=tf.string),
        'likes': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0)
    }

    # Define context features for each image in the set
    for i in range(number_set_images):
        context_features[f"{image_feature}/{i}"] = tf.io.FixedLenFeature([], dtype=tf.string, default_value='')

    sequence_features = {
        image_index: tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        caption_feature: tf.io.VarLenFeature(dtype=tf.int64)
    }

    context, sequence = tf.io.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features
    )

    set_id = context[set_id]
    likes = context['likes']

    encoded_images = [context[f"{image_feature}/{i}"] for i in range(number_set_images)]

    captions = tf.sparse.to_dense(sequence[caption_feature])
    image_ids = sequence[image_index]

    return set_id, encoded_images, image_ids, captions, likes


def prefetch_input_data(file_pattern, batch_size, is_training, values_per_shard, input_queue_capacity_factor=16, num_reader_threads=1):
    """Prefetches string values from disk into an input queue."""

    # Get list of data files using tf.io.gfile.glob
    data_files = tf.io.gfile.glob(file_pattern)
    if not data_files:
        raise FileNotFoundError(f"Found no input files matching {file_pattern}")
    else:
        print(f"Prefetching values from {len(data_files)} files matching {file_pattern}")

    dataset = tf.data.Dataset.from_tensor_slices(data_files)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(data_files))

    def _read_file(filename):
        return tf.data.TFRecordDataset(filename)

    dataset = dataset.interleave(
        _read_file,
        cycle_length=num_reader_threads,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(buffer_size=values_per_shard * input_queue_capacity_factor)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def batch_with_dynamic_pad(images_and_captions, batch_size, add_summaries=True):
    """Batches input images and captions with dynamic padding."""
    def process_element(set_id, images, image_ids, captions, likes):
        image_seq_length = tf.shape(image_ids)[0]
        input_length = image_seq_length

        cap_indicator = tf.cast(tf.not_equal(captions, tf.zeros_like(captions)), tf.int32)
        indicator = tf.ones([input_length], dtype=tf.int32)
        loss_indicator = tf.ones([image_seq_length], dtype=tf.int32)
        images = tf.stack(images)

        return set_id, images, indicator, loss_indicator, image_ids, captions, cap_indicator, likes
    
    dataset = images_and_captions.map(process_element)

    # Create a dataset from the list of batched data
    # dataset = tf.data.Dataset.from_generator(
    #     lambda: iter(enqueue_list),
    #     output_signature=(
    #         tf.TensorSpec(shape=(), dtype=tf.string),  # set_id
    #         tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # images (dynamic height/width)
    #         tf.TensorSpec(shape=(None,), dtype=tf.int32),  # mask
    #         tf.TensorSpec(shape=(None,), dtype=tf.int32),  # loss_mask
    #         tf.TensorSpec(shape=(None,), dtype=tf.int64),  # image_ids
    #         tf.TensorSpec(shape=(None,), dtype=tf.int64),  # captions
    #         tf.TensorSpec(shape=(None,), dtype=tf.int32),  # cap_mask
    #         tf.TensorSpec(shape=(), dtype=tf.int64)  # likes
    #     )
    # )
    
    # Add a debug map to print shapes of the dataset elements
    def print_shapes(*args):
        for item in args:
            print(f"Shape of item: {tf.shape(item)}")
        return args  # Make sure to return the elements unchanged

    # Map the debug print function to print shapes of each batch
    dataset = dataset.map(print_shapes)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [],                         # set_id (scalar)
            [None, None, None, 3],      # images (dynamic height/width, 3 channels)
            [None],                     # mask (variable-length)
            [None],                     # loss_mask (variable-length)
            [None],                     # image_ids (variable-length)
            [None, None],               # captions (variable-length)
            [None, None],               # cap_mask (variable-length)
            []                          # likes (scalar)
        ),
        padding_values=(
            "",                         # Padding value for set_id
            0.0,                        # Padding value for images
            0,                          # Padding value for mask
            0,                          # Padding value for loss_mask
            tf.constant(0, dtype=tf.int64),                          # Padding value for image_ids
            tf.constant(0, dtype=tf.int64),                          # Padding value for captions
            tf.constant(0, dtype=tf.int32),                          # Padding value for cap_mask
            tf.constant(0, dtype=tf.int64)                           # Padding value for likes
        )
    )
    
    # Optionally, add summaries for caption length
    # if add_summaries:
    #     dataset = dataset.map(lambda x: {
    #         **x,
    #         "caption_length": tf.reduce_sum(x["mask"], axis=1)
    #     })

    return dataset
