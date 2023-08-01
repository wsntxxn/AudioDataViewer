from flask import Flask, render_template, request, jsonify
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import os
import librosa
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from data_reader import READER_REGISTRY

app = Flask(__name__)

Path("static").mkdir(exist_ok=True)


datasets = list(READER_REGISTRY.keys())

reader = None


def plot_spectrogram(waveform, sr):
    # stft = librosa.core.stft(y=waveform, n_fft=1024, 
        # hop_length=10 * sr // 1000, window='hann', center=True)

    plt.figure(figsize=(12, 3))
    # plt.matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=2048, hop_length=10 * sr // 1000, n_mels=64)
    log_melspec = librosa.amplitude_to_db(spectrogram)
    plt.imshow(log_melspec, origin='lower', aspect='auto', cmap='jet')

    plt.ylabel('Frequency bins')
    duration = len(waveform) / sr
    ticks = [int(x) for x in np.arange(0, log_melspec.shape[1] * 6 / 5, int(log_melspec.shape[1] / 5))]
    labels = np.arange(0, duration * 6 / 5, duration / 5)
    labels = ["{:.2f}".format(x) for x in labels]
    plt.xticks(ticks[:6], labels[:6])
    plt.xlabel("Seconds")
    plt.title("Log Mel-spectrogram")
    plt.savefig("static/spectrogram.jpg", bbox_inches="tight", dpi=100)


def get_audio_by_id(audio_id):
    waveform, sr, labels = reader.get_audio_by_id(audio_id)
    plot_spectrogram(waveform, sr)
    return waveform, sr, labels


@app.route("/show", methods=["POST"])
def index():
    if "audio_id" in request.form:
        audio_id = request.form["audio_id"]
        waveform, sample_rate, labels = get_audio_by_id(audio_id)
        file_path = "static/tmp.wav"
        sf.write(file_path, waveform, sample_rate)
        return jsonify({"audio_id": audio_id, "labels": labels})
    elif "dataset" in request.form:
        global reader
        dataset = request.form["dataset"]
        reader = READER_REGISTRY[dataset]()
        return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def select():
    return render_template("select.html", datasets=datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=10086)
    args = parser.parse_args()
    app.run(debug=True, host="0.0.0.0", port=args.port)

