from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import path

def mp3_to_wav(source, destination):
    # files  
    src = source
    dst = destination

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

def split(filepath):
    sound = AudioSegment.from_wav(filepath)
    dBFS = sound.dBFS
    chunks = split_on_silence(sound, 
        min_silence_len = 500,
        silence_thresh = dBFS-16)
    return chunks

def split_multi_chunk(source, dest_path):
    audio_chunks = split(source)
    for i, chunk in enumerate(audio_chunks):
        out_file = "chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(path.join(dest_path, out_file), format="wav")