import { useState, useEffect } from "react";
import { io } from "socket.io-client";
import "../styles/AudioUploader.css";

const socket = io("http://127.0.0.1:5000"); // Connect to Flask backend

const Home = () => {
    const [audio, setAudio] = useState(null);
    const [progress, setProgress] = useState([]);
    const [complete, setComplete] = useState(false);
    const [downloadUrl, setDownloadUrl] = useState("");

    useEffect(() => {
        socket.on("progress_update", (data) => {
            setProgress((prev) => [...prev, `✅ ${data.message}`]);
        });

        socket.on("process_complete", (data) => {
            setComplete(true);
            setDownloadUrl(data.url);
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

        setProgress([]); // Reset progress
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
            <div className="card">
                <input 
                    type="file" 
                    accept="audio/mp3" 
                    onChange={handleFileChange} 
                    className="file-input"
                />
                <button 
                    onClick={handleUpload} 
                    className="upload-button"
                >
                    Upload & Process
                </button>
                
                {progress.map((msg, index) => (
                    <p key={index} className="message">{msg}</p>
                ))}
                {complete && 
                <button onClick={handleDownloadClick} className="download-button">
                    Download Censored Audio
                </button>}
            </div>
        </div>
    );
}

export default Home;
