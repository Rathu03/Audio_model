import cv2
import tempfile
import os
from pydub import AudioSegment

def extract_audio(video_path, output_audio_path):
    # Create a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    # Open the video with OpenCV
    cap = cv2.VideoCapture(video_path)

    # Extract audio using PyDub
    audio = AudioSegment.from_file(video_path)
    audio.export(temp_wav.name, format="wav")

    # Convert to mp3 (or save directly as wav)
    final_audio = AudioSegment.from_wav(temp_wav.name)
    final_audio.export(output_audio_path, format="mp3")

    # Clean up
    temp_wav.close()
    os.unlink(temp_wav.name)
    cap.release()
    print(f"Audio extracted to {output_audio_path}")

# Example usage
extract_audio("Video_model/Inputs/test.mp4", "Extracted_Audio/output.mp3")
