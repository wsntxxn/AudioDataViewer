import numpy as np
import pandas as pd
import h5py


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
