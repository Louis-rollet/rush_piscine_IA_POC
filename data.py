import os
import re

import numpy as np
import pandas as pd
import pydub
import torch.nn as nn
import torch
import torchaudio
from tqdm import tqdm


class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels,
                                                              win_length=win_length, hop_length=hop_length)

    def forward(self, x):
        x = self.transform(x)
        x = np.log(x + 1e-14)
        return x


class TextProcess:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_table = {char: i for i, char in enumerate(self.alphabet)}
        self.index_map = {i: char for i, char in enumerate(self.alphabet)}

    def encode(self, text):
        text = re.sub(r'[éèê]', 'e', text)
        text = re.sub(r'[àâ]', 'a', text)
        text = re.sub(r'î', 'i', text)
        text = re.sub(r'ô', 'o', text)
        text = re.sub(r'û', 'u', text)
        text = re.sub(r'ç', 'c', text)
        text = text.lower()
        text = re.sub(r"[^a-z ']", '', text)
        return [self.char_table[char] for char in text]

    def decode(self, indices):
        return ''.join([self.index_map[i] for i in indices])


class SpecAugment(nn.Module):

    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


class DataBase:
    def __init__(self, path, alphabet, limit=None):
        self.path = path
        self.limit = limit
        self.text_process = TextProcess(alphabet)
        self.transform = LogMelSpec()
        self.train_csv = pd.read_csv(os.path.join(self.path, 'train.tsv'), sep='\t')
        self.test_csv = pd.read_csv(os.path.join(self.path, 'test.tsv'), sep='\t')

        self.data = {'data': [], 'label': []}
        self.train_data = None
        self.load()

    def load(self):
        for i, file in tqdm(self.train_csv['path'].items()):
            if self.limit is not None and i > self.limit:
                break
            path = os.path.join(self.path, 'clips', file + '.mp3')
            audio = torchaudio.load(path)[0]
            text = self.text_process.encode(self.train_csv['sentence'][i])
            self.data['data'].append(self.transform(audio))
            self.data['label'].append(text)
        # self.train_data = pd.DataFrame(self.data)


alphabet = "' abcdefghijklmnopqrstuvwxyz"
db = DataBase('dataset/fr/', alphabet, limit=5000)
print("DONE !")
