import cv2
import librosa
import math
import json 

import os
import base64
import shutil

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

from codeFeature.feature_extractor_cnn import *
from codeFeature.file_split import *

dataset_path = r"./genres_original"
json_path = r"./class_xls/data.json"
image_music = "./image_music"

model_path = "./class_xls/model_cnn.json"
weight_path = "./class_xls/model_cnn_weights.h5"

dest_chunk = './musics/chunk'
data_music_path = "./musics/wav"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

adam = optimizers.Adam(lr=1e-4)
genres, nb_train_samples = GetGenre(dataset_path)
predict_list = []

def load_model(path, weigth):
    return load__cnn_model(path, weigth)
model_load = load_model(model_path, weight_path)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def mp3_to_wav(source, destination):
    # files  
    src = source
    dst = destination

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

def remove_file_dir(dir):
    for files in os.listdir(dir):
        path_file = os.path.join(dir, files)
        try:
            shutil.rmtree(path_file)
        except OSError:
            os.remove(path_file)

menu = ["Home","Play Music","About me"]
choice = st.sidebar.selectbox("What ", menu)

if choice =="Home":
    st.header("Hello Music Genre Classification")
    st.markdown(''' 
        - Music Genre Classification – Automatically classify different musical genres. In this tutorial we are going to develop a deep learning project to automatically classify different musical genres from audio files. We will classify these audio files using their low-level features of frequency and time domain. For this project we need a dataset of audio tracks having similar size and similar frequency range. GTZAN genre classification dataset is the most recommended dataset for the music genre classification project and it was collected for this task only.
        ![image info](image_music/Description/Flow.png)
        
            ``` 
            <!-- BLOG-POST-LIST:START -->
            <!-- BLOG-POST-LIST:END -->
            ```
        
        ''')

elif choice =="Play Music":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader('Choose a mp3 file that you extracted from the work site')
    uploaded_file = st.file_uploader('Select')
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        
        with open(os.path.join(data_music_path,uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())         
        st.success("Saved File")
        sound = AudioSegment.from_mp3((os.path.join(data_music_path,uploaded_file.name)))
        if uploaded_file.name:
            wav_file_name = uploaded_file.name[:-4]+'.wav'
            wav_file_path = os.path.join(data_music_path,wav_file_name)
            sound.export(wav_file_path, format="wav")

            st.audio(wav_file_path, format='audio/wav')

            remove_file_dir(dest_chunk)
            #split file into many path
            split_multi_chunk(wav_file_path, dest_chunk)

            predict_list = []
            for i, (dirpath, _, filenames) in enumerate(os.walk(dest_chunk)):
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    one_pitch = predict_cnn(file_path, model_load, num_segments=10)
                    if one_pitch != "none":
                        predict_list.append(genres[one_pitch])
                    else:
                        print('none predict')

            image_list_genres = []
            for genre_image in predict_list:
                image_list_genres.append(os.path.join(image_music, genre_image+'.png'))
            st.image(image_list_genres, caption= 'Music Genres', width=400)