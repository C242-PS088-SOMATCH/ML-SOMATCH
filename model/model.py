import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import InceptionV3
import os


# Function to parse TFRecord files and extract features
def _parse_function(proto):
    # Define your tfrecord structure as you described in the feature map
    feature_map = {
        'set_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'set_url': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'likes': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'views': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'images/0': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/1': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/3': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/4': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/5': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/6': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'images/7': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'caption': tf.io.VarLenFeature(tf.string),
        'caption_ids': tf.io.VarLenFeature(tf.int64),
        'image_idx': tf.io.VarLenFeature(tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(proto, feature_map)
    
    images = [tf.io.decode_jpeg(parsed_features[f'images/{i}']) for i in range(8)]
    caption = tf.sparse.to_dense(parsed_features['caption'])
    caption_ids = tf.sparse.to_dense(parsed_features['caption_ids'])
    
    return images, caption, caption_ids

def create_model(vocab_size, embedding_dim, lstm_units=512):
    # 1. InceptionV3 Model for Feature Extraction
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the layers of InceptionV3 for transfer learning
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # 2. Visual-Semantic Embedding: Projecting images and captions into a shared space
    image_embedding = layers.Dense(lstm_units, activation='relu')(x)

    # 3. Bi-LSTM for Captions
    caption_input = layers.Input(shape=(None,), dtype=tf.int32)  # Input for captions
    caption_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(caption_input)
    caption_lstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(caption_embedding)

    # Ensure both embeddings have the same shape before combining them
    combined_embedding = layers.Add()([image_embedding, caption_lstm])

    # 4. Compatibility Score Prediction
    compatibility_output = layers.Dense(1, activation='sigmoid', name='compatibility')(combined_embedding)
    
    # Define the complete model
    model = models.Model(inputs=[base_model.input, caption_input], outputs=compatibility_output)
    
    return model


def build_inception_v3_model(vocab_size, embedding_dim, lstm_units=512):
    # Define inputs: 8 images and the sequence of captions
    image_inputs = [layers.Input(shape=(299, 299, 3)) for _ in range(8)]  # 8 image inputs
    caption_input = layers.Input(shape=(None,), dtype=tf.int32)  # Caption input
    
    # Initialize the image feature extraction model
    image_features = []
    for img_input in image_inputs:
        img_feature = create_model(vocab_size, embedding_dim, lstm_units)([img_input, caption_input])  # Pass both inputs
        image_features.append(img_feature)
    
    # Compatibility Prediction based on image features and caption embeddings
    compatibility_scores = []
    for feature in image_features:
        compatibility_score = layers.Dense(1)(feature)  # Measure compatibility score for each image
        compatibility_scores.append(compatibility_score)
    
    # Combine the compatibility scores into a final output
    final_compatibility = layers.Concatenate()(compatibility_scores)
    
    model = models.Model(inputs=[image_inputs, caption_input], outputs=final_compatibility)
    
    return model

# # Example usage:
# vocab_size = 5000  # Example vocab size (adjust based on your dataset)
# embedding_dim = 256  # Embedding size for both images and captions

# model = build_inception_v3_model(vocab_size, embedding_dim)
# model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# # Now you can use the model and train it with your dataset
