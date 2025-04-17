import cv2
import tempfile
import os
from pydub import AudioSegment

def extract_audio(video_path, output_audio_path):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    cap = cv2.VideoCapture(video_path)

    audio = AudioSegment.from_file(video_path)
    audio.export(temp_wav.name, format="wav")

    final_audio = AudioSegment.from_wav(temp_wav.name)
    final_audio.export(output_audio_path, format="mp3")

    temp_wav.close()
    os.unlink(temp_wav.name)
    cap.release()
    print(f"Audio extracted to {output_audio_path}")

extract_audio("Video_model/Inputs/test.mp4", "Extracted_Audio/output.mp3")
