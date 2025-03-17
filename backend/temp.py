from flask import Flask, request, send_file
from flask_socketio import SocketIO
import time
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "Censored_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@socketio.on("upload_audio")
def handle_audio(data):
    audio_file = data["audio"]
    #file_name = os.path.splitext(os.path.basename(audio_file))[0]
    # Save the audio file
    #print(audio_file)
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.mp3")
    with open(audio_path, "wb") as f:
        f.write(audio_file)

    print("Audio received and saved:", audio_path)

    # Simulate audio processing (replace with actual ML/DL)
    for i in range(5):  
        time.sleep(1)  # Simulating processing delay
        socketio.emit("progress_update", {"message": f"Processing step {i + 1}"})

    processed_audio_path = os.path.join(PROCESSED_FOLDER,"audio_censored.mp3")

    socketio.emit("process_complete", {"message": "Processing complete!","url": "/download_audio"}) 

@app.route("/download_audio")
def download_audio():
    processed_audio_path = os.path.join(PROCESSED_FOLDER, "audio_censored.mp3")
    
    if os.path.exists(processed_audio_path):
        return send_file(processed_audio_path, as_attachment=True)
    else:
        return {"error": "Processed audio not found"}, 404

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
