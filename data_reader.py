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


class ClassificationReader:

    def __init__(self, wav_csv, label, label_column, sr):
        wav_df = pd.read_csv(wav_csv, sep="\t")
        label_df = pd.read_csv(label, sep="\t")

        self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
        self.aid_to_labels = dict(zip(label_df["audio_id"], label_df[label_column]))
        self.sr = sr

    def get_audio_by_id(self, audio_id):
        with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
            waveform = np.array(hf[audio_id], dtype=np.float32)
        labels = self.aid_to_labels[audio_id].split(";")
        return waveform, self.sr, labels


class AudioCaptionReader:

    def __init__(self, wav_csv, label, sr) -> None:
        wav_df = pd.read_csv(wav_csv, sep="\t")
        data = json.load(open(label))

        self.aid_to_labels = {}
        for item in data["audios"]:
            audio_id = item["audio_id"]
            self.aid_to_labels[audio_id] = []
            for cap_item in item["captions"]:
                self.aid_to_labels[audio_id].append(cap_item["caption"])

        self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
        self.sr = 32000

    def get_audio_by_id(self, audio_id):
        with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
            waveform = np.array(hf[audio_id], dtype=np.float32)
        labels = self.aid_to_labels[audio_id]
        return waveform, self.sr, labels


@register_reader("AudioSet")
class AudioSetReader(ClassificationReader):

    def __init__(self) -> None:
        super().__init__("/mnt/lustre/sjtu/home/xnx98/work/AudioTagging/audioset_tagging/data/audioset/full_train/waveform.csv",
                         "/mnt/lustre/sjtu/home/xnx98/work/AudioTagging/audioset_tagging/data/audioset/full_train/label.csv",
                         "event_labels",
                         32000)


@register_reader("ESC50")
class Esc50Reader(ClassificationReader):

    def __init__(self) -> None:
        super().__init__("/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/esc50/waveform_32k.csv",
                         "/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/esc50/label.csv",
                         "category",
                         32000)


@register_reader("Clotho")
class ClothoReader(AudioCaptionReader):

    def __init__(self) -> None:
        super().__init__("/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/clotho_v2/dev/waveform_32k.csv",
                         "/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/clotho_v2/dev/text.json",
                         32000)


@register_reader("AudioCaps")
class AudioCapsReader(AudioCaptionReader):
    
    def __init__(self) -> None:
        super().__init__("/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/audiocaps/all/waveform.csv",
                         "/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/audiocaps/all/text.json",
                         32000)

    def get_audio_by_id(self, audio_id):
        try:
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf[audio_id], dtype=np.float32)
        except:
            with h5py.File(self.aid_to_h5[audio_id], "r") as hf:
                waveform = np.array(hf["Y" + audio_id + ".wav"], dtype=np.float32)
        labels = self.aid_to_labels[audio_id]
        return waveform, self.sr, labels
    

@register_reader("UrbanSound8K")
class Us8KReader(ClassificationReader):

    def __init__(self) -> None:
        super().__init__("/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/us8k/waveform_32k.csv",
                         "/mnt/lustre/sjtu/home/xnx98/work/AudioClassify/data/us8k/label.csv",
                         "class",
                         32000)


@register_reader("MACS")
class MacsReader(AudioCaptionReader):

    def __init__(self) -> None:
        wav_csv = "/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/macs/waveform.csv"
        label = "/mnt/lustre/sjtu/home/xnx98/work/AudioCaption/data/macs/text.json"
        sr = 32000
        super().__init__(wav_csv, label, sr)
