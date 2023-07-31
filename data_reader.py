import json
import numpy as np
import pandas as pd
import h5py


READER_REGISTRY = {}


def register_reader(name):
    def decorator(cls):
        READER_REGISTRY[name] = cls
        return cls
    return decorator


@register_reader("AudioSet")
class AudioSetReader:

    def __init__(self) -> None:
        wav_df = pd.read_csv("/mnt/lustre/sjtu/home/xnx98/work/AudioTagging/audioset_tagging/data/audioset/full_train/waveform.csv", sep="\t")
        label_df = pd.read_csv("/mnt/lustre/sjtu/home/xnx98/work/AudioTagging/audioset_tagging/data/audioset/full_train/label.csv", sep="\t")

        self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
        self.aid_to_labels = dict(zip(label_df["audio_id"], label_df["event_labels"]))
        self.sr = 32000

    def get_audio_by_id(self, audio_id):
        try:
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        except:
            audio_id = "Y" + audio_id + ".wav"
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        labels = self.aid_to_labels[audio_id].split(";")
        return waveform, self.sr, labels


@register_reader("ESC50")
class Esc50Reader:

    def __init__(self) -> None:
        wav_df = pd.read_csv("/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/esc50/waveform_32k.csv", sep="\t")
        label_df = pd.read_csv("/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/esc50/label.csv", sep="\t")

        self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
        self.aid_to_labels = dict(zip(label_df["audio_id"], label_df["category"]))
        self.sr = 32000

    def get_audio_by_id(self, audio_id):
        try:
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        except KeyError:
            audio_id = "Y" + audio_id + ".wav"
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        labels = [self.aid_to_labels[audio_id]]
        return waveform, self.sr, labels


@register_reader("Clotho")
class ClothoReader:

    def __init__(self) -> None:
        wav_df = pd.read_csv("/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/clotho_v2/dev/waveform_32k.csv", sep="\t")
        label = json.load(open("/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/clotho_v2/dev/text.json"))

        self.aid_to_labels = {}
        for item in label["audios"]:
            audio_id = item["audio_id"]
            self.aid_to_labels[audio_id] = []
            for cap_item in item["captions"]:
                self.aid_to_labels[audio_id].append(cap_item["caption"])

        self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
        self.sr = 32000

    def get_audio_by_id(self, audio_id):
        try:
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        except KeyError:
            audio_id = "Y" + audio_id + ".wav"
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        labels = self.aid_to_labels[audio_id]
        return waveform, self.sr, labels
