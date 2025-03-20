'use client';

import { useState, useEffect } from 'react';
import { io } from 'socket.io-client';

const socket = io('http://127.0.0.1:5000');

const VideoUploader = () => {
    const [video, setVideo] = useState(null);
    const [progress, setProgress] = useState([]);
    const [detections, setDetections] = useState([]);

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
            socket.emit('upload_video_file', { video: reader.result });
        };
    };

    return (
        <div className='container'>
            <h1 className='header'>Video Censoring Process</h1>

            {/* Hide the upload section if detections exist */}
            {detections.length === 0 && (
                <div className='card'>
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

            {progress.map((msg, index) => (
                <p key={index} className='message'>{msg}</p>
            ))}

            {detections.length > 0 && (
                <div className="detections">
                    <h3>Filtered Detections:</h3>

                    {/* Wrapper with scrolling */}
                    <div className="table-wrapper">
                        <table className="detection-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Timestamp</th>
                                    <th>Class</th>
                                    <th>X1</th>
                                    <th>Y1</th>
                                    <th>X2</th>
                                    <th>Y2</th>
                                </tr>
                            </thead>
                            <tbody>
                                {detections.map((detection, index) => (
                                    <tr key={index}>
                                        <td>{detection.model}</td>
                                        <td>{detection.timestamp}</td>
                                        <td>{detection.class}</td>
                                        <td>{detection.x1}</td>
                                        <td>{detection.y1}</td>
                                        <td>{detection.x2}</td>
                                        <td>{detection.y2}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

        </div>
    );
};

export default VideoUploader;
