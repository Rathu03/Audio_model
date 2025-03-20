import { useState, useEffect } from "react";
import { io } from "socket.io-client";
import "../styles/Main.css";

const socket = io("http://127.0.0.1:5000"); // Connect to Flask backend

const AudioUploader = () => {
    const [audio, setAudio] = useState(null);
    const [progress, setProgress] = useState([]);
    const [complete, setComplete] = useState(false);
    const [downloadUrl, setDownloadUrl] = useState("");
    const [comp, setComp] = useState(false)

    useEffect(() => {
        socket.on("progress_update", (data) => {
            setProgress((prev) => [...prev, `✅ ${data.message}`]);
        });

        socket.on("process_complete", (data) => {
            setComplete(true);
            if(data.url != "nothing"){
                setDownloadUrl(data.url)
                setComp(true)
            }
            else{
                setDownloadUrl("")
                setComp(false)
            }
        
        });

        return () => {
            socket.off("progress_update");
            socket.off("process_complete");
        };
    }, []);

    const handleFileChange = (event) => {
        setAudio(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!audio) return;

        setProgress([]); 
        setComplete(false);
        setDownloadUrl("")

        const reader = new FileReader();
        reader.readAsArrayBuffer(audio);
        reader.onload = () => {
            socket.emit("upload_audio", { audio: reader.result });
        };
    };


    const handleDownloadClick = () => {
        if(downloadUrl){
            window.open("http://127.0.0.1:5000/download_audio", "_blank")
        }
    }

    return (
        <div className="container">
            <h1 className="header">Audio Censoring Process</h1>
            <div className="card">
                <label htmlFor="file-upload" className="file-label">
                    Choose Audio File
                </label>
                <input 
                    id="file-upload"
                    type="file" 
                    accept="audio/mp3" 
                    onChange={handleFileChange} 
                    className="file-input"
                />
                {audio && <p className="file-name">📂 {audio.name}</p>}
                <button 
                    onClick={handleUpload} 
                    className="upload-button"
                >
                    Upload & Process
                </button>
                
                {progress.map((msg, index) => (
                    <p key={index} className="message">{msg}</p>
                ))}
    
                {comp && 
                <>
                
                <audio controls className="audio-player">
                    <source src={"http://127.0.0.1:5000/download_audio"} type="audio/mp3" />
                    Your browser does not support the audio element.
                </audio>

                <button onClick={handleDownloadClick} className="download-button">
                    Download Censored Audio
                </button>
                </>}
                
            </div>
        </div>
    );
}

export default AudioUploader;
