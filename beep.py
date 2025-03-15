from pydub import AudioSegment
from pydub.generators import Sine
import os

input_audio = "./Audio/vadachennai1.mp3"
timestamps = [[ 1.14, 1.58]]

def create_beep(duration_ms):
    return Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-5)

def beep(input_audio,timestamps):
    audio = AudioSegment.from_file(input_audio)  

    for start, end in timestamps:
        st = int(start*1000)
        en = int(end*1000)

        print(st,en)

        if en > len(audio):
            en = len(audio)
        if st >= len(audio):
            continue
        
        duration = en - st
        beep_segment = create_beep(duration)
        
        silent_segment = AudioSegment.silent(duration=duration)
        audio = audio[:st] + silent_segment + audio[en:]
        
        audio = audio.overlay(beep_segment, position=st)

    file_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_audio = f"./Censored_audio/{file_name}_censored.mp3"

    audio.export(output_audio, format="mp3")
    print(f"Beeped audio saved as {output_audio}")

beep(input_audio,timestamps)
