import subprocess
import os

UPLOAD_FOLDER1 = "uploads1"
os.makedirs(UPLOAD_FOLDER1, exist_ok=True)
PROCESSED_FOLDER1 = "Censored_video"
os.makedirs(PROCESSED_FOLDER1,exist_ok=True)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "Censored_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def download_audio_video():
    original_video_path = os.path.join(PROCESSED_FOLDER1, "processed_video.mp4")
    fixed_video_path = os.path.join(PROCESSED_FOLDER1, "processed_fixed.mp4")
    audio_path = os.path.join(PROCESSED_FOLDER, "audio_censored.mp3")  # Assuming the audio file is in the same folder
    output_video_path = os.path.join(PROCESSED_FOLDER1, "output.mp4")  # Final merged output video path

    if os.path.exists(original_video_path):
        # Re-encode the original video first
        try:
            result = subprocess.run([
                'ffmpeg', '-y',  # -y to overwrite if exists
                '-i', original_video_path,
                '-vcodec', 'libx264',  # Video codec
                '-acodec', 'aac',      # Audio codec
                fixed_video_path       # Output path for the fixed video
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Log the output and error of the FFmpeg command
            print(f"FFmpeg re-encoding output: {result.stdout.decode()}")
            print(f"FFmpeg re-encoding error: {result.stderr.decode()}")
        except subprocess.CalledProcessError as e:
            return {"error": f"ffmpeg failed during re-encoding: {e}, {e.stderr}"}, 500

        # Now merge the re-encoded video and audio
        try:
            result = subprocess.run([
                'ffmpeg', '-y',  # -y to overwrite if exists
                '-i', fixed_video_path,      # Input the re-encoded video
                '-i', audio_path,            # Input the audio file
                '-c:v', 'copy',              # Copy video codec (no re-encoding)
                '-c:a', 'aac',               # Audio codec to AAC
                '-strict', 'experimental',   # Allow experimental codecs (needed for AAC)
                '-shortest',                 # Ensure the shortest stream is used
                output_video_path            # Final output path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Log the output and error of the FFmpeg command
            print(f"FFmpeg merge output: {result.stdout.decode()}")
            print(f"FFmpeg merge error: {result.stderr.decode()}")

        except subprocess.CalledProcessError as e:
            return {"error": f"ffmpeg failed during merging: {e}, {e.stderr}"}, 500
    else:
        return {"error": "Processed video not found"}, 404

download_audio_video()