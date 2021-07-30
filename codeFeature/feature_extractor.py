import os
import csv
import librosa
import numpy as np
import ntpath
import librosa
import librosa.display
import IPython.display as ipd

eps = np.spacing(1)

def FileCheck(fn):
    try:
        librosa.load(fn, mono=True, duration=30)
        return 1
    except:
        print (f"Error: File {fn} does not appear to exist.")
        return 0

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def GetFeatureSection(songname, genre, s, y, sr):
    
    TOTAL_SAMPLES = 29 * 22050
    NUM_SLICES = 10
    SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)
    start_sample = SAMPLES_PER_SLICE * s
    end_sample = start_sample + SAMPLES_PER_SLICE
    # print(SAMPLES_PER_SLICE)
    y = y[start_sample:end_sample]
    length = 66149
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma_stft.mean()
    chroma_stft_var = chroma_stft.var()
    rmse = librosa.feature.rms(y=y)
    rms_mean = rmse.mean()
    rms_var = rmse.var()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent_mean = spec_cent.mean()
    spec_cent_var = spec_cent.var()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_mean = spec_bw.mean()
    spec_bw_var = spec_bw.var()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    zcr_var = zcr.var()
    y_harm, y_perc = librosa.effects.hpss(y=y)
    harmony_mean = y_harm.mean()
    harmony_var = y_harm.var()
    perceptr_mean = y_perc.var()
    perceptr_var = y_perc.mean()
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    songname = songname+str(s)
    to_append = f'{songname} {chroma_stft_mean} {chroma_stft_var} {rms_mean} {rms_var} {spec_cent_mean} {spec_cent_var} {spec_bw_mean} {spec_bw_var} {rolloff_mean} {rolloff_var} {zcr_mean} {zcr_var} {harmony_mean} {harmony_var} {perceptr_mean} {perceptr_var} {tempo}'
    for e in mfcc:
        to_append += f' {np.mean(e)} {np.var(e)}'
    to_append += f' {genre}'
    return to_append

def extract_Feature_audio(songname, genre='_'):
    data_predict = []
    if FileCheck(songname)!=1:
        pass
    y, sr = librosa.load(songname, mono=True)
    # , duration=30
    duration = librosa.get_duration(y=y, sr=sr)
    # print(duration)
    sections = int(duration/30)
    # print(sections)
    file_name = path_leaf(songname)
    for i in range(sections*10):
        trypre = GetFeatureSection(file_name, genre, i, y, sr).split()
        trypre = trypre[1:-1]
        trypre = np.array(trypre, dtype = float)
        data_predict.append(trypre)
    return data_predict

def gen_header_csv():
    header = 'filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    return header

def extract_Feature_audio_toCSV(genre_music_path, csv_file, genres):
    header = gen_header_csv()
    file = open(csv_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        for genre in genres:
            for filename in os.listdir('{0}/{1}'.format(genre_music_path, genre)):
                # print(genre_music_path,filename)
                songname = '{0}/{1}/{2}'.format(genre_music_path, genre, filename)
                # print(genre_music_path,songname)
                if FileCheck(songname)!=1:
                    continue
                y, sr = librosa.load(songname, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                sections = int(duration/30)
                file_name = path_leaf(songname)
                data_predict = []
                for i in range(sections*10):
                    trypre = GetFeatureSection(file_name, genre, i, y, sr).split()
                    trypre = trypre[1:-1]
                    trypre = np.array(trypre, dtype = float)
                    data_predict.append(trypre)
                # file = open(csv_file, 'w', newline='')
                # with file:
                write_data = np.array(data_predict)
                writer = csv.writer(file, delimiter=',')
                writer.writerows(write_data)
    file.close()
            

