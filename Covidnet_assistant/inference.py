import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import os
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

from data import feature_extraction

@dataclass
class InferenceConfig:
    sample_rate: int
    n_mfcc: int
    n_mels: int
    n_fft: int
    length_ms: int
    mean_size: int
    use_segmentation: bool
    use_normalization: bool
    use_trim: bool
    trim_ref: int

    def __post_init__(self):
        assert self.n_mfcc <= self.n_mels

def add_noise(audio, noise_level=0.000001):
    noise = np.random.normal(0, noise_level, audio.shape[0])
    return noise + audio

def process_audio(audio_path, config, save_plot_path):
    audio = librosa.load(audio_path.resolve(), sr=config.sample_rate)[0]
    _, processed_audio = feature_extraction((None, audio, config), save_plot_path=save_plot_path)
    return np.expand_dims(processed_audio, axis=(0, -1))

def inference_h5(model_path, audio):
    model = keras.models.load_model(model_path)
    y_pred = model.predict(audio)[0]
    return np.argmax(y_pred), y_pred[0], y_pred[1]

def inference_graph(model_path, audio):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        writer = tf.summary.FileWriter(os.path.join(model_path, 'graph'), sess.graph)
        y_pred = sess.run("output/Softmax:0", feed_dict={"input:0": audio})[0]
        writer.close()
    return np.argmax(y_pred), y_pred[0], y_pred[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for covid cough models.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to audio file.")
    parser.add_argument("--model_type", type=str, default="graph", help="h5/graph")
    parser.add_argument("--n_mels", type=int, default=32, help="Number of mel.")
    parser.add_argument("--n_mfcc", type=int, default=32, help="Number of mfcc.")
    parser.add_argument("--n_fft", type=int, default=2048, help="Number of fft.")
    parser.add_argument(
        "--length_ms", type=int, default=7000, help="Max length of audio."
    )
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate.")
    parser.add_argument(
        "--mean_size", type=int, default=2, help="Mean size to use when preprocessing."
    )
    parser.add_argument(
        "--use_segmentation",
        action="store_true",
        default=False,
        help="Use segmentation or not.",
    )
    parser.add_argument(
        "--use_normalization",
        action="store_true",
        default=False,
        help="Use normalization or not.",
    )
    parser.add_argument(
        "--use_trim",
        action="store_true",
        default=False,
        help="Use trim or not.",
    )
    parser.add_argument("--trim_ref", type=int, default=20, help="ref when trimming audios.")
    parser.add_argument("--save_plot_location", type=str, default="inference_plot", help="Path to model.")

    args = parser.parse_args()
    inference_config = InferenceConfig(
        **{
            k.name: getattr(args, k.name)
            for k in fields(InferenceConfig)
            if k.name in vars(args)
        }
    )
    if args.model_type == "h5": 
        inference = inference_h5
    elif args.model_type == "graph": 
        inference = inference_graph
    else:
        print("Unrecognized model type, should be either h5 or graph!")
        exit(1) 

    inference_result, neg_conf, pos_conf = inference(args.model, process_audio(args.audio, inference_config, args.save_plot_location))
    
    if inference_result == 0:
        print("Negative;", "Confidence:", neg_conf)
    else:
        print("Positive;", "Confidence:", pos_conf)
