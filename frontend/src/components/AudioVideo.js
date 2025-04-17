'use client';

import { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useNavigate } from 'react-router-dom';

const socket = io('http://127.0.0.1:5000');

const AudioVideo = () => {
    const [video, setVideo] = useState(null);
    const [progress, setProgress] = useState([]);
    const [detections, setDetections] = useState([]);

    const navigate = useNavigate()

    useEffect(() => {
        socket.on('video_process_complete', (data) => {
            setProgress((prev) => [...prev, data.message]);
            setDetections(data.detections);
        });

        socket.on('error', (data) => {
            console.error('Error:', data.message);
        });

        return () => {
            socket.off('video_process_complete');
            socket.off('error');
        };
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('video/')) {
            setVideo(file);
        } else {
            alert('Please select a valid video format.');
        }
    };

    const handleUpload = () => {
        if (!video) return;

        setProgress([]);
        setDetections([]);

        const reader = new FileReader();
        reader.readAsArrayBuffer(video);
        reader.onload = () => {
            socket.emit('upload_audio_video', { video: reader.result });
        };
    };

    return (
        <div className='container'>
            {/* <h1 className='header'>Video Censoring Process</h1> */}
            {detections.length === 0 && (
                <div className='card'>
                    <h1 style={{ marginBottom: '30px' }}>🔊🎥 Audio & Video Moderation</h1>
                    <label htmlFor='file-upload-video' className='file-label'>
                        Choose Video File
                    </label>
                    <input
                        id='file-upload-video'
                        type='file'
                        accept='video/*'
                        onChange={handleFileChange}
                        className='file-input'
                    />
                    {video && <p className='file-name'>📂 {video.name}</p>}

                    <button onClick={handleUpload} className='upload-button'>
                        Upload & Process
                    </button>
                </div>
            )}

            <button className="download-button" onClick={() => navigate("../")} style={{width:"13%",marginLeft:"80%"}}>
                Back
            </button>

        </div>
    );
}

export default AudioVideo
