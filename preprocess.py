import whisper
import librosa
import noisereduce as nr
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import torch


def preprocess_audio(input_audio):
    y,sr = librosa.load(input_audio,sr=None)

    reduced_noise = nr.reduce_noise(y=y,sr=sr)

    output_wav = "cleaned_audio.wav"
    sf.write(output_wav,reduced_noise,sr)

    return output_wav

input_audio_path = "./audio/santhanam.mp3"
cleaned_audio = preprocess_audio(input_audio_path)
print("Cleaned audio generated successfully")