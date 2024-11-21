import tensorflow as tf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))

from model import build_inception_v3_model, _parse_function


# Parameters
BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = (299, 299)  # InceptionV3 input size
VOCAB_SIZE = 5000        # Example vocab size (adjust based on your dataset)
EMBEDDING_DIM = 256      # Embedding size for both images and captions
LSTM_UNITS = 512         # Number of hidden units for the LSTM
TRAIN_TFRECORD_PATH = "data/tf_records/train-no-dup-00001-of-00010"  # Path to your training TFRecord file
VALID_TFRECORD_PATH = "data/tf_records/valid-no-dup-00001-of-00005"  # Path to your validation TFRecord file

# Load TFRecord Dataset
def load_tfrecord_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)  # Apply the parsing function
    dataset = dataset.shuffle(buffer_size=10000)  # Shuffle the dataset
    dataset = dataset.batch(BATCH_SIZE)  # Batch the dataset
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch to improve performance
    return dataset

# Preprocess the images (resize and normalize)
def preprocess_image(image):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.inception_v3.preprocess_input(image)  # Normalize the image
    return image

# Load train and validation datasets
train_dataset = load_tfrecord_dataset(TRAIN_TFRECORD_PATH)
valid_dataset = load_tfrecord_dataset(VALID_TFRECORD_PATH)

# Define the model
model = build_inception_v3_model(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS)
model.summary()  # Check the model summary to verify its structure

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',  # Adjust loss based on your task (here, binary compatibility prediction)
    metrics=['accuracy']
)

# Model Checkpoint Callback to save the best model based on validation loss
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'model_checkpoint.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min', 
    verbose=1
)

# Early stopping callback to stop training when the validation loss doesn't improve
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,  # Number of epochs with no improvement before stopping
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset, 
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Save the final model
model.save('final_model.h5')

# Optionally, plot training history for analysis (if needed)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss over Epochs')
plt.legend(loc='upper left')
plt.show()
