import os
import json
import numpy as np
import pickle as pkl
from sklearn import metrics
import tensorflow as tf
import argparse
import polyvore.bi_lstm as polyvore_model
import configuration


# Setup argument parser to replace tf.flags
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="", help="Model checkpoint file or directory containing a model checkpoint file.")
parser.add_argument("--label_file", type=str, default="", help="Txt file containing test outfits.")
parser.add_argument("--feature_file", type=str, default="", help="Files containing image features")
parser.add_argument("--rnn_type", type=str, default="", help="Type of RNN.")
parser.add_argument("--result_file", type=str, default="", help="File to store the results.")
parser.add_argument("--direction", type=int, default=2, help="2: bidirectional; 1: forward only; -1: backward only.")
FLAGS = parser.parse_args()

def run_compatibility_inference(image_seqs, test_feat, num_lstm_units, model):
    emb_seqs = test_feat[image_seqs, :]
    num_images = float(len(image_seqs))
    if FLAGS.rnn_type == "lstm":
        zero_state = np.zeros([1, 2 * num_lstm_units])
    else:
        zero_state = np.zeros([1, num_lstm_units])

    f_score = 0
    b_score = 0

    if FLAGS.direction != -1:
        # Forward RNN.
        outputs = []
        input_feed = np.reshape(emb_seqs[0], [1, -1])
        lstm_state, lstm_output = model.predict(input_feed)  # Use Keras predict
        outputs.append(lstm_output)

        # Run remaining steps.
        for step in range(int(num_images) - 1):
            input_feed = np.reshape(emb_seqs[step + 1], [1, -1])
            lstm_state, lstm_output = model.predict(input_feed)
            outputs.append(lstm_output)

        s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(test_feat)))
        f_score = model.evaluate(s)

    if FLAGS.direction != 1:
        # Backward RNN.
        outputs = []
        input_feed = np.reshape(emb_seqs[-1], [1, -1])
        lstm_state, lstm_output = model.predict(input_feed)
        outputs.append(lstm_output)
        for step in range(int(num_images) - 1):
            input_feed = np.reshape(emb_seqs[int(num_images) - 2 - step], [1, -1])
            lstm_state, lstm_output = model.predict(input_feed)
            outputs.append(lstm_output)

        s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(test_feat)))
        b_score = model.evaluate(s)

    return [f_score, b_score]


def main():
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model_config = configuration.ModelConfig()
        model_config.rnn_type = FLAGS.rnn_type
        model = polyvore_model.PolyvoreModel(model_config, mode="inference")
        model.build()
        model.load_weights(FLAGS.checkpoint_path)  # Load weights directly

        # Load pre-computed image features.
        with open(FLAGS.feature_file, "rb") as f:
            test_data = pkl.load(f)
        test_ids = list(test_data.keys())
        test_feat = np.zeros((len(test_ids) + 1, len(test_data[test_ids[0]]["image_rnn_feat"])))

        # test_feat has one more zero vector as the representation of END of RNN prediction.
        for i, test_id in enumerate(test_ids):
            test_feat[i] = test_data[test_id]["image_rnn_feat"]

        all_f_scores = []
        all_b_scores = []
        all_scores = []
        all_labels = []
        testset = open(FLAGS.label_file).read().splitlines()
        k = 0
        for test_outfit in testset:
            k += 1
            if k % 100 == 0:
                print(f"Finish {k} outfits.")
            image_seqs = [test_ids.index(test_image) for test_image in test_outfit.split()[1:]]

            f_score, b_score = run_compatibility_inference(image_seqs, test_feat, model_config.num_lstm_units, model)

            all_f_scores.append(f_score)
            all_b_scores.append(b_score)
            all_scores.append(f_score + b_score)
            all_labels.append(int(test_outfit[0]))

        # Calculate AUC and AP
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
        print(f"Compatibility AUC: {metrics.auc(fpr, tpr):f} for {len(all_labels)} outfits")

        # Save the results
        with open(FLAGS.result_file, "wb") as f:
            pkl.dump({"all_labels": all_labels, "all_f_scores": all_f_scores, "all_b_scores": all_b_scores}, f)


if __name__ == "__main__":
    main()
