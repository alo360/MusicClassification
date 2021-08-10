import os

import math
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import librosa
import librosa.display
import IPython.display as ipd

from tensorflow.keras import optimizers
# from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import model_from_json

adam = optimizers.Adam(learning_rate=1e-4)

# dataset_path = r"../genres_original"
# json_path = r"../class_xls/data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def GetGenre(dataset_path):
    label_names = [item for item in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, item))]
    nb_train_samples = sum([len(files) for _, _, files in os.walk(dataset_path)])
    return label_names, nb_train_samples

def FileCheck(fn):
    try:
        librosa.load(fn, mono=True, duration=30)
        return 1
    except:
        print(f"Error: File {fn} does not appear to exist.")
        return 0


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_feature_mfcc(filenames, n_mfcc=13, n_fft=2048,
                     hop_length=512, num_segments=10):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK/num_segments)  # ps = per segment
    expected_vects_ps = math.ceil(samples_ps/hop_length)

    # process files for specific genre
    if FileCheck(filenames) != 1:
        # As librosa only read files <1Mb
        return data
    else:
        # load audio file
        signal, sr = librosa.load(filenames, sr=SAMPLE_RATE, duration=30)
        if librosa.get_duration(y=signal, sr=sr) <30:
            return data
        for s in range(num_segments):
            start_sample = samples_ps * s
            finish_sample = start_sample + samples_ps

            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length)

            mfcc = mfcc.T

            # store mfcc if it has expected length
            if len(mfcc) == expected_vects_ps:
                data["mfcc"].append(mfcc.tolist())
                # data["labels"].append(i-1)
                # print(f"{filenames}, segment: {s+1}")
    return data


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK/num_segments)  # ps = per segment
    expected_vects_ps = math.ceil(samples_ps/hop_length)

    # loop through all the genres
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")

            # process files for specific genre
            for f in filenames:
                if(f == str("jazz.00054.wav")):
                    # As librosa only read files <1Mb
                    continue
                else:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    for s in range(num_segments):
                        start_sample = samples_ps * s
                        finish_sample = start_sample + samples_ps

                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)

                        mfcc = mfcc.T

                        # store mfcc if it has expected length
                        if len(mfcc) == expected_vects_ps:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print(f"{file_path}, segment: {s+1}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Convert list to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def prepare_dataset(test_size, validation_size, json_path):
    X, y = load_data(json_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_CNN_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (2, 2), activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (1, 1), activation="relu"))
    model.add(MaxPool2D((1, 1), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer=adam,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # model.summary()
    return model

def predict_cnn(filepath, model, num_segments=10):
    extractaudio_features = get_feature_mfcc(filepath, num_segments=num_segments)
    if len(extractaudio_features["mfcc"]) == 0:
        return 'none'
    extractaudio_features = np.array(extractaudio_features["mfcc"])
    extractaudio_features = extractaudio_features[..., np.newaxis]
    prediction = model.predict(extractaudio_features)
    data_predict = []
    for i in range(prediction.shape[0]):
        data_predict.append(prediction[1])
    predict_genre = np.argmax(data_predict)
    return predict_genre


def load__cnn_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model