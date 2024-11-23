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
        raise ValueError(f"Found no input files matching {file_pattern}")
    else:
        print(f"Prefetching values from {len(data_files)} files matching {file_pattern}")

    # Create a dataset from TFRecord files
    dataset = tf.data.TFRecordDataset(data_files)

    # Map the parse function to decode each record
    dataset = dataset.map(lambda serialized: parse_sequence_example(
        serialized,
        set_id='set_id',  # Adjust as per context feature name
        image_feature='image_feature',  # Adjust as per image feature name
        image_index='image_index',  # Adjust as per image index feature name
        caption_feature='caption_feature',  # Adjust as per caption feature name
        number_set_images=5  # Adjust the number of images per set
    ))

    # Shuffle the dataset if in training mode
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    return dataset


def batch_with_dynamic_pad(images_and_captions, batch_size, queue_capacity, add_summaries=True):
    """Batches input images and captions, with dynamic padding."""

    # Convert the list of images and captions into a tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: images_and_captions,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # set_id
            tf.TensorSpec(shape=(None,), dtype=tf.string),  # images (list of strings)
            tf.TensorSpec(shape=(None,), dtype=tf.int64),  # image_ids
            tf.TensorSpec(shape=(None,), dtype=tf.int64),  # captions
            tf.TensorSpec(shape=(), dtype=tf.int64)  # likes
        )
    )

    # Dynamic padding will be applied to images and captions
    padded_dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [],  # set_id
            [None],  # images
            [None],  # image_ids
            [None],  # captions
            []  # likes
        ),
        padding_values=(
            '',  # set_id padding
            '',  # image padding
            0,   # image_ids padding
            0,   # caption padding
            0    # likes padding
        )
    )

    # Optionally add summary information about the captions
    if add_summaries:
        padded_dataset = padded_dataset.map(lambda set_id, images, image_ids, captions, likes: (
            set_id,
            images,
            image_ids,
            captions,
            likes
        ))

    return padded_dataset
