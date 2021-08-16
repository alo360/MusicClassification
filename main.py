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
import tubestats as tb
from tubestats.youtube_search import *
from tubestats.setAPIkey import YT_API_KEY
from tubestats.youtube_api import YouTubeAPI
from tubestats.youtube_data import YouTubeData
# import tubestats.youtube_search as ys
from pathlib import Path
from matplotlib import cm
from datetime import datetime, timedelta

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

def date_slider(date_end=datetime.today()):
        date_start, date_end = st.slider(
                'Select date range to include:',
                min_value=first_video_date, # first video
                max_value=last_video_date, #value for date_end
                value=(first_video_date , last_video_date), #same as min value
                step=timedelta(days=2),
                format='YYYY-MM-DD',
                key=999)
        return date_start, date_end

def remove_file_dir(dir):
    for files in os.listdir(dir):
        path_file = os.path.join(dir, files)
        try:
            shutil.rmtree(path_file)
        except OSError:
            os.remove(path_file)

@st.cache
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



menu = ["Home","Feature Extractors","Play Music","About me"]
choice = st.sidebar.selectbox("What ", menu)



if choice =="Home":
    st.header("Hello Music Genre Classification")
    # gif_html = get_img_with_href('image_music\Description\Flow.png', 'https://docs.streamlit.io')
    # st.markdown(gif_html, unsafe_allow_html=True)
    
    intro_markdown = read_markdown_file("./image_music/Description/md01.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

elif choice =="Feature Extractors":
    st.subheader('Choose a mp3 file that you extracted MFCC features and plot it')
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
            audio, sfreq = librosa.load(wav_file_path)
            # time = np.arange(0, len(audio))/sfreq
            mfcc_data = librosa.feature.mfcc(audio[0:66150],n_fft=2048)
            
            st.write("Plot chart MFCC in 3s of "+uploaded_file.name)
            librosa.display.specshow(mfcc_data, x_axis='time')
            plt.colorbar()
            plt.tight_layout()
            plt.title('mfcc')
            st.pyplot(plt)
            
            st.write("Plot chart song signal "+uploaded_file.name)
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(211)
            ax1.set_title("Plot chart song signal")
            ax1.set_xlabel('time')
            ax1.set_ylabel('Amptitude')
            librosa.display.waveplot(audio)
            st.pyplot(fig)



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

                selection = ''
                if len(predict_list)>0:
                    selection = st.selectbox('Select', predict_list)
                    if st.button("Classification Musics!") and selection !='':
                        search_result = search_key_word(selection)
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
                        
                        st.header('Videos')
                        """
                        Below is a graph plotting the views of each video over time. Please note:
                        - colour represents the like and dislike
                        - size represents the number of views.
                        - a natural log axis is applied to the view count due to its 'viral' nature
                        """
                        first_video_date = df['snippet.publishedAt_REFORMATED'].min().to_pydatetime()
                        last_video_date = df['snippet.publishedAt_REFORMATED'].max().to_pydatetime()

                        date_start, date_end = date_slider()
                        transformed_df = youtuber_data.transform_dataframe(date_start=date_start, date_end=date_end) 
                        c = youtuber_data.scatter_all_videos(transformed_df)
                        st.altair_chart(c, use_container_width=True)

                        st.subheader('Videos by Time Difference')
                        """
                        This looks at the time difference between the current video and the previous video.
                        """
                        time_df = youtuber_data.time_difference_calculate(df=transformed_df)
                        time_diff = youtuber_data.list_time_difference_ranked(df=time_df)
                        st.altair_chart(youtuber_data.time_difference_plot(df=time_df), use_container_width=True)

                        quantiles = youtuber_data.time_difference_statistics(df=time_df)
                        st.subheader('Time Difference Statistics:')
                        st.markdown('25th Percentile: `' + '{:0.1f}'.format(quantiles[0.25]) + '` days')
                        st.markdown('Median: `' + '{:0.1f}'.format(quantiles[0.50]) + '` days')
                        st.markdown('75th Percentile: `' + '{:0.1f}'.format(quantiles[0.75]) + '` days')
                        st.markdown('Longest Hiatus: `' + '{:0.1f}'.format(quantiles[1.]) + '` days')
                    
                        vid_list = youtuber_data.greatest_time_difference_video(time_df)
                        st.subheader('Longest Hiatus:')
                        st.video('https://www.youtube.com/watch?v=' + str(vid_list['greatest']))
                        prev_col, next_col = st.columns(2)
                        with prev_col:
                            st.subheader('Previous:')
                            st.video('https://www.youtube.com/watch?v=' + str(vid_list['prev']))
                        with next_col:
                            st.subheader('Next:')
                            st.video('https://www.youtube.com/watch?v=' + str(vid_list['_next']))
                        st.write(time_diff)

                        def display_vid_links(most_viewed_info):
                            st.write('Here are links to the videos:')
                            titles = most_viewed_info['title']
                            links = most_viewed_info['link']
                            for i in range(len(titles)):
                                title = str(titles[i])
                                link = 'https://www.youtube.com/watch?v=' + str(links[i])
                                if i == 0:
                                    st.write(str(i+1) + '. ' + title)
                                    st.video(data=link)
                                else:
                                    st.markdown(str(i+1) + '. ' + '[' + title +']' + '(' + link + ')')

                        st.header('Most Popular Videos')
                        """
                        Hypothesis: view count indicates well performing videos. The content is engaging enough and liked to be recommended and viewed more often to other viewers.
                        """
                        most_viewed_info = youtuber_data.most_viewed_videos(df=transformed_df)
                        st.write(most_viewed_info['preserved_df'])
                        display_vid_links(most_viewed_info)

                        #dislike_num = st.slider('Number of videos', 5, 20, key=0)
                        st.header('Most Unpopular Videos')
                        """
                        Remaining a hypothesis, people actively show their digust for a video by hitting dislike video. Hence, we are provided with a like-dislike ratio. We also have the sum to ensure we have enough likes/dislikes for fair comparison.
                        """
                        most_disliked_info = youtuber_data.most_disliked_videos(df=transformed_df)
                        st.write(most_disliked_info['preserved_df'])
                        display_vid_links(most_disliked_info) 
                        
                        st.header('List of Video')
                        """
                        List of videos and all relevant features.
                        """
                        st.write(df)