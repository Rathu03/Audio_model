'use client';

import { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useNavigate } from 'react-router-dom';
import { Puff } from 'react-loading-icons';

const socket = io('http://127.0.0.1:5000');

const AudioVideo = () => {
    const [video, setVideo] = useState(null);
    const [progress1, setProgress1] = useState([])
    const [progress2, setProgress2] = useState([]);
    const [detections, setDetections] = useState([]);
    const [comp1, setComp1] = useState(false)
    const [load,setLoad] = useState(false)
    const [downloadUrl, setDownloadUrl] = useState("")

    const navigate = useNavigate()

    useEffect(() => {

        socket.on("progress_update1", (data) => {
            setProgress1((prev) => [...prev, `✅ ${data.message}`]);
            setLoad(true)
        });

        // socket.on("process_complete1", (data) => {
        //     if(data.url != "nothing"){
        //         setComp(true)
        //     }
        //     else{
        //         setComp(false)
        //     }
        
        // });

        socket.on('video_process_complete1', (data) => {
            setProgress2((prev) => [...prev, data.message]);
            setDetections(data.detections);
            setDownloadUrl(data.url)
            setLoad(false)
            setComp1(!comp1)
            console.log(detections)
        });

        socket.on('error', (data) => {
            console.error('Error:', data.message);
        });

        return () => {
            socket.off("process_update1")
            socket.off("process_complete1")
            socket.off('video_process_complete1');
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

        setProgress1([]);
        setProgress2([])
        setDetections([]);
        setLoad(true)

        const reader = new FileReader();
        reader.readAsArrayBuffer(video);
        reader.onload = () => {
            socket.emit('upload_audio_video', { video: reader.result });
        };
        alert("Video Uploaded Successfully")
    };


    const overlayStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backdropFilter: 'blur(5px)',
        backgroundColor: 'rgba(0, 0, 0, 0.2)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 9999
    };

    return (
        <div className='container' >
            {/* <h1 className='header'>Video Censoring Process</h1> */}
            
            {load && (
                <div style={overlayStyle}>
                    <Puff stroke="#98ff98" />
                </div>
                
            )}
            
            <div className='card'>
            {!comp1 && (
                <>
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
                    
                </>
                
                
                
            )}

            
                
                    

                    {/* {progress1.map((msg,index) => (
                        <p key={index} className="message">{msg}</p>
                    ))} */}

                    

                    {comp1 && (
                        <>
                        
                        {/* {progress2.map((msg,index) => (
                                <p key={index} className='message'>{msg}</p>
                            ))} */}
                        
                            <div className="detections" style={{marginTop:"200px"}}>
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
                                <video controls className="video-player" style={{marginTop:"20px"}} width="640" height="360">
                                    <source src={"http://127.0.0.1:5000/download_audio_video"} type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
    
                            </div>
                            </>
                    )}

                    

                        

                    

                </div>
            

            <div style={{display:"flex"}}>
                <button className="download-button" onClick={() => window.location.reload()} style={{width:"60%",marginLeft:"100%",textAlign:"center"}}>
                    Refresh
                </button>
                <button className="download-button" onClick={() => navigate("../")} style={{width:"50%",marginLeft:"30%"}}>
                    Back
                </button>
            </div>

        </div>
    );
}

export default AudioVideo
