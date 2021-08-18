import librosa

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
import tubestats as tb
from tubestats.youtube_search import *
from tubestats.youtube_api import YouTubeAPI
from tubestats.youtube_data import YouTubeData
# import tubestats.youtube_search as ys
from pathlib import Path
from matplotlib import cm

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

genres, nb_train_samples = GetGenre(dataset_path)
# predict_list = []
genres_np = np.array(genres)
list_genres = []
predict_np = []

def read_markdown_file(markdown_file):
    text = Path(markdown_file).read_text()
    text = text.replace("![Screenshot](", "![Screenshot](http://localhost:8501/")
    return text

@st.cache(allow_output_mutation=True)
def load_model(path, weigth):
    return load__cnn_model(path, weigth)

model_load = load_model(model_path, weight_path)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def remove_file_dir(dir):
    for files in os.listdir(dir):
        path_file = os.path.join(dir, files)
        try:
            shutil.rmtree(path_file)
        except OSError:
            os.remove(path_file)

@st.cache
def store_predict(predict):
    return predict

@st.cache(suppress_st_warning=True)
def fetch_data(user_input):
    youtuber_data = YouTubeData(user_input)
    return youtuber_data

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

menu = ["Home","Feature Extractors","Play Music"]
choice = st.sidebar.selectbox("Main Menu", menu)

if choice =="Home":
    st.header("Hello Music Genre Classification")
    # gif_html = get_img_with_href('image_music\Description\Flow.png', 'https://docs.streamlit.io')
    # st.markdown(gif_html, unsafe_allow_html=True)
    
    intro_markdown = read_markdown_file("./image_music/Description/md01.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

elif choice =="Feature Extractors":
    st.subheader('Choose a mp3 file that you extracted MFCC features and plot it')
    uploaded_file = st.file_uploader('Select', key="4")
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
            audio, sfreq = librosa.load(wav_file_path)
            # time = np.arange(0, len(audio))/sfreq

            st.write("Plot chart song signal "+uploaded_file.name)
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(211)
            ax1.set_title("Plot chart song signal")
            ax1.set_xlabel('time')
            ax1.set_ylabel('Amptitude')
            librosa.display.waveplot(audio)
            st.pyplot(fig)

            mfcc_data = librosa.feature.mfcc(audio[0:66150],n_fft=2048, n_mfcc=13, hop_length=512)
            
            st.write("Plot chart MFCC in 3s of "+uploaded_file.name)
            librosa.display.specshow(mfcc_data, x_axis='time')
            plt.colorbar()
            plt.tight_layout()
            plt.title('mfcc')
            st.pyplot(plt)
            st.write(mfcc_data.T)
            


elif choice =="Play Music":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.image("image_music/Description/Music_Genre_Feature.jpg", width=800)
    # st.subheader('Choose a mp3 file that you extracted from the work site')
    uploaded_file = st.file_uploader('Choose a mp3 file that you extracted from the work site', key="1")
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        
        with open(os.path.join(data_music_path,uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())         
        # st.success("Saved File")
        sound = AudioSegment.from_mp3((os.path.join(data_music_path,uploaded_file.name)))
        if uploaded_file.name:
            wav_file_name = uploaded_file.name[:-4]+'.wav'
            wav_file_path = os.path.join(data_music_path,wav_file_name)
            sound.export(wav_file_path, format="wav")

            st.audio(wav_file_path, format='audio/wav')
        
        guess_genres = []
        # Using the "with" syntax
        with st.form(key='form01'):
            guess_genres = st.multiselect("We Bet you can guess music genres based On the heard song", options=genres, key="2")
            submit_button = st.form_submit_button(label='Classification Musics!')
            
            # Classification = st.button("Classification Musics!",key='3')
            # if Classification:
        if len(guess_genres):
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
                    # else:
                    #     print('none predict')

            progress_bar.progress(70)
            status_text.text('Ready to classify .....')

            image_list_genres = []
            for genre_image in predict_list:
                image_list_genres.append(os.path.join(image_music, genre_image+'.png'))
            st.image(image_list_genres, width=160)

            predict_np = np.array(predict_list)
            number_match = [x for x in predict_np if x in np.array(guess_genres)]
            
            list_genres = predict_list.copy()

            progress_bar.progress(80)
            status_text.text('Ready to classify .....')
            if len(number_match)>0:
                st.write("You guess correct of ",len(number_match))
                st.balloons()
            progress_bar.progress(100)
            status_text.text('Completed All Task!')

        # selection_genre = st.empty()
            selection_genre = st.radio('Select the genres', options=list_genres, key='4')
            if selection_genre:
                search_result = search_key_word(selection_genre)
                search_result = search_result['items'][0]['id']['videoId']
                search_result = "https://www.youtube.com/watch?v="+search_result
                st.write(search_result)
                youtuber_data = fetch_data(search_result)
                df = youtuber_data.dataframe()

                st.header(youtuber_data.channel_name())
                img_col, stat_col = st.columns(2)
                with img_col:
                    st.image(youtuber_data.thumbnail_url())
                with stat_col:  
                    st.subheader('Quick Statistics')
                    st.markdown('Total Number of Videos: `' + '{:,}'.format(int(youtuber_data.video_count())) + '`')
                    st.markdown('Join Date: `' + str(youtuber_data.start_date()) + '`')
                    st.markdown('Total View Count:  `' + '{:,}'.format(int(youtuber_data.total_channel_views())) + '`')
                    st.markdown('Total Comments: `' + '{:,}'.format(int(youtuber_data.total_comments())) + '`')
                    st.markdown('Total Watch Time: `' + str(youtuber_data.total_watchtime()) + '`')
                st.write(youtuber_data.channel_description())