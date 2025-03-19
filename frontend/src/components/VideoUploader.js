import { useState, useEffect } from "react";
import { io } from "socket.io-client";
import "../styles/Main.css";

const socket = io("http://127.0.0.1:5000");

const VideoUploader = () => {
    const [video, setVideo] = useState(null);
    const [progress, setProgress] = useState([]);
    const [complete, setComplete] = useState(false);
    const [downloadUrl, setDownloadUrl] = useState("");
    const [comp, setComp] = useState(false);

    useEffect(() => {
        socket.on("video_progress_update", (data) => {
            setProgress((prev) => [...prev, `📹 ${data.message}`]);
        });

        socket.on("video_process_complete", (data) => {
            setComplete(true);
            if (data.url !== "nothing") {
                setDownloadUrl(data.url);
                setComp(true);
            } else {
                setDownloadUrl("");
                setComp(false);
            }
        });

        return () => {
            socket.off("video_progress_update");
            socket.off("video_process_complete");
        };
    }, []);

    const handleFileChange = (event) => {
        setVideo(event.target.files[0]);
    };

    const handleUpload = () => {
        if (!video) return;

        setProgress([]);
        setComplete(false);
        setDownloadUrl("");

        const reader = new FileReader();
        reader.readAsArrayBuffer(video);
        reader.onload = () => {
            socket.emit("upload_video", { video: reader.result });
        };
    };

    const handleDownloadClick = () => {
        if (downloadUrl) {
            window.open("http://127.0.0.1:5000/download_video", "_blank");
        }
    };

    return (
        <div className="uploader-card">
            <h2>Video Upload</h2>
            <input type="file" accept="video/mp4" onChange={handleFileChange} />
            {video && <p>🎥 {video.name}</p>}

            <button onClick={handleUpload} className="upload-button">
                Upload & Process
            </button>

            {progress.map((msg, index) => (
                <p key={index} className="message">{msg}</p>
            ))}

            {comp && (
                <>
                    <video controls className="video-player" width="500">
                        <source src="http://127.0.0.1:5000/download_video" type="video/mp4" />
                        Your browser does not support the video element.
                    </video>

                    <button onClick={handleDownloadClick} className="download-button">
                        Download Processed Video
                    </button>
                </>
            )}
        </div>
    );
};

export default VideoUploader;
