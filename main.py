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
from pathlib import Path



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
genres_np = np.array(genres)

def read_markdown_file(markdown_file):
    text = Path(markdown_file).read_text()
    text = text.replace("![Screenshot](", "![Screenshot](http://localhost:8501/")
    return text

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

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code



menu = ["Home","Play Music","About me"]
choice = st.sidebar.selectbox("What ", menu)



if choice =="Home":
    st.header("Hello Music Genre Classification")
    # gif_html = get_img_with_href('image_music\Description\Flow.png', 'https://docs.streamlit.io')
    # st.markdown(gif_html, unsafe_allow_html=True)
    
    intro_markdown = read_markdown_file("./image_music/Description/md01.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

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
            guess_genres = st.multiselect("We Bet you can guess music genres based On the heard song", genres)
            if st.button("Classification Musics!"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                remove_file_dir(dest_chunk)
                #split file into many path
                split_multi_chunk(wav_file_path, dest_chunk)
                
                progress_bar.progress(10)
                status_text.text('Begin to classify is: %s percent' % 10)


                predict_list = []
                for i, (dirpath, _, filenames) in enumerate(os.walk(dest_chunk)):
                    for f in filenames:
                        file_path = os.path.join(dirpath, f)
                        one_pitch = predict_cnn(file_path, model_load, num_segments=10)
                        if one_pitch != "none":
                            predict_list.append(genres[one_pitch])
                        else:
                            print('none predict')

                progress_bar.progress(70)
                status_text.text('Ready to classify .....')

                image_list_genres = []
                for genre_image in predict_list:
                    image_list_genres.append(os.path.join(image_music, genre_image+'.png'))
                st.image(image_list_genres, width=160)

                predict_np = np.array(predict_list)
                number_match = [x for x in predict_np if x in np.array(guess_genres)]

                progress_bar.progress(80)
                status_text.text('Ready to classify .....')
                if len(number_match)>0:
                    st.write("You guess correct of ",len(number_match))
                    st.balloons()
                progress_bar.progress(100)
                status_text.text('Completed All Task!')
