import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

DATA_PATH = "datasets/"

def get_labels(path=DATA_PATH):
    labels = os.listdir(path+'sesler')
    label_indices = np.arange(0, len(labels))
    category = to_categorical(label_indices)
    return  labels, label_indices, category

def wav2mfcc(file_path, dim1, dim2):
    wave, sr = librosa.load(file_path, sr=None, mono=True)
    # 83142 byte (record size) / 4 / 2 = 10392
    mfcc = librosa.feature.mfcc(wave, n_mfcc=dim1)
    if dim2>mfcc.shape[1]:
        pad_width = dim2 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :dim2]
    return mfcc

def get_train_test(dim1, dim2):
    labels, indices, hot_vector = get_labels(DATA_PATH)
    # labels = ['asagi', 'sol', 'sag', 'yukari']
    # [0, 1, 2, 3]
    # [
    # [1 0 0 0]
    # [0 1 0 0]
    # [0 0 1 0]
    # [0 0 0 1]
    # ]
    indices = 0
    x = []
    y = []
    for label in labels:
        wav_files = ['datasets/sesler/' +label+'/' + wavfile for wavfile in os.listdir('datasets/sesler/'+label) ]
        for wav_file in wav_files:
            mfcc = wav2mfcc(wav_file, dim1, dim2)
            x.append(mfcc)
            y.append(hot_vector[indices])
        indices = indices + 1
    x = np.array(x)
    y = np.array(y)

    return train_test_split(x, y, test_size=int(len(x)*0.3), shuffle=True)

