import json
import tensorflow as tf
import numpy as np
import pickle as pkl
import argparse

import configuration
import polyvore_model_bi as polyvore_model

# Setup argument parser for TensorFlow 2.x
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="", help="Model checkpoint file or directory containing a model checkpoint file.")
parser.add_argument("--json_file", type=str, default="", help="Json file containing questions and answers.")
parser.add_argument("--feature_file", type=str, default="", help="pkl files containing the features.")
parser.add_argument("--rnn_type", type=str, default="lstm", help="Type of RNN.")
parser.add_argument("--result_file", type=str, default="", help="File to store the results.")
parser.add_argument("--direction", type=int, default=2, help="2: bidirectional; 1: forward only; -1: backward only; 0: Average pooling no RNN.")
FLAGS = parser.parse_args()

def run_question_inference(model, question, test_ids, test_feat, test_rnn_feat, num_lstm_units):
    question_ids = []
    answer_ids = []
    for q in question["question"]:
        try:
            question_ids.append(test_ids.index(q))
        except:
            return [], []
    
    for a in question["answers"]:
        try:
            answer_ids.append(test_ids.index(a))
        except:
            return [], []
        
    blank_posi = question["blank_position"]
    
    # Average pooling of the VSE embeddings
    question_emb = np.reshape(np.mean(test_feat[question_ids], 0), [1,-1])
    q_emb = question_emb / np.linalg.norm(question_emb, axis=1)[:, np.newaxis]
    a_emb = (test_feat[answer_ids] / np.linalg.norm(test_feat[answer_ids], axis=1)[:, np.newaxis])
    vse_score = (np.dot(q_emb, np.transpose(a_emb)) + 1) / 2  # scale to [0,1]
    
    if FLAGS.direction == 0:
        # Only use VSE
        predicted_answer = np.argsort(-vse_score)[0]
        return vse_score, predicted_answer
        
    if FLAGS.rnn_type == "lstm":
        # LSTM has two states.
        zero_state = np.zeros([1, 2 * num_lstm_units])
    else:
        zero_state = np.zeros([1, num_lstm_units])
    
    # Blank is the last item.
    if blank_posi == len(question_ids) + 1:
        if FLAGS.direction == -1:
            return [], []
        # Only do forward rnn
        input_feed = np.reshape(test_rnn_feat[question_ids[0]], [1,-1])
        lstm_state, lstm_output = model.predict(input_feed)
        
        for step in range(len(question_ids)-1):
            input_feed = np.reshape(test_rnn_feat[question_ids[step + 1]], [1,-1])
            lstm_state, lstm_output = model.predict(input_feed)
            
        # Search in answers
        rnn_score = np.exp(np.dot(lstm_output, np.transpose(test_rnn_feat[answer_ids])))
        rnn_score = rnn_score / np.sum(rnn_score)
        
    # Blank is the first item
    elif blank_posi == 1:
        if FLAGS.direction == 1:
            return [], []
        # only do backward rnn
        input_feed = np.reshape(test_rnn_feat[question_ids[-1]], [1,-1])
        lstm_state, lstm_output = model.predict(input_feed)
        
        for step in range(len(question_ids)-1):
            input_feed = np.reshape(test_rnn_feat[question_ids[-step-2]], [1,-1])
            lstm_state, lstm_output = model.predict(input_feed)
        
        rnn_score = np.exp(np.dot(lstm_output, np.transpose(test_rnn_feat[answer_ids])))
        rnn_score = rnn_score / np.sum(rnn_score)
    
    # Blank is in the middle.
    else:
        # Do bidirectional rnn.
        # Forward:
        input_feed = np.reshape(test_rnn_feat[question_ids[0]], [1,-1])
        lstm_state, lstm_output = model.predict(input_feed)
        
        for step in range(blank_posi - 2):
            input_feed = np.reshape(test_rnn_feat[question_ids[step + 1]], [1,-1])
            lstm_state, lstm_output = model.predict(input_feed)
            
        # Search in answers.
        f_softmax = np.exp(np.dot(lstm_output, np.transpose(test_rnn_feat[answer_ids])))
        
        # Backward:
        input_feed = np.reshape(test_rnn_feat[question_ids[-1]], [1,-1])
        lstm_state, lstm_output = model.predict(input_feed)
        
        for step in range(len(question_ids)-blank_posi):
            input_feed = np.reshape(test_rnn_feat[question_ids[-step-2]], [1,-1])
            lstm_state, lstm_output = model.predict(input_feed)

        b_softmax = np.exp(np.dot(lstm_output, np.transpose(test_rnn_feat[answer_ids])))
        
        if FLAGS.direction == 2:
            rnn_score = (f_softmax / np.sum(f_softmax) + b_softmax / np.sum(b_softmax)) / 2
        elif FLAGS.direction == 1:
            rnn_score = f_softmax / np.sum(f_softmax)
        else:
            rnn_score = b_softmax / np.sum(b_softmax)

    predicted_answer = np.argsort(-rnn_score)[0]
    return rnn_score, predicted_answer

def main():
    top_k = 4  # Print the top_k accuracy.
    true_pred = np.zeros(top_k)

    # Load pre-computed image features.
    with open(FLAGS.feature_file, "rb") as f:
        test_data = pkl.load(f)
    test_ids = list(test_data.keys())
    test_feat = np.zeros((len(test_ids), len(test_data[test_ids[0]]["image_feat"])))
    test_rnn_feat = np.zeros((len(test_ids), len(test_data[test_ids[0]]["image_rnn_feat"])))

    for i, test_id in enumerate(test_ids):
        test_feat[i] = test_data[test_id]["image_feat"]
        test_rnn_feat[i] = test_data[test_id]["image_rnn_feat"]

    model_config = configuration.ModelConfig()
    model_config.rnn_type = FLAGS.rnn_type
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    model.load_weights(FLAGS.checkpoint_path)  # Load model weights

    questions = json.load(open(FLAGS.json_file))

    all_pred = []
    set_ids = []
    all_scores = []
    
    for question in questions:
        score, pred = run_question_inference(model, question, test_ids, test_feat, test_rnn_feat, model_config.num_lstm_units)
        if pred != []:
            all_pred.append(pred)
            all_scores.append(score)
            set_ids.append(question["question"][0].split("_")[0])
            
            # Calculate top-k accuracy.
            for i in range(top_k):
                if 0 in pred[:i+1]:
                    true_pred[i] += 1

    # Print all top-k accuracy.
    for i in range(top_k):
        print(f"Top {i+1} Accuracy: {true_pred[i]} correct answers in {len(all_pred)} valid questions.")
        print(f"Accuracy: {true_pred[i] / len(all_pred):f}")

    s = np.empty((len(all_scores),), dtype=np.object)
    for i in range(len(all_scores)):
        s[i] = all_scores[i]

    with open(FLAGS.result_file, "wb") as f:
        pkl.dump({"set_ids": set_ids, "pred": all_pred, "score": s}, f)

if __name__ == "__main__":
    main()
