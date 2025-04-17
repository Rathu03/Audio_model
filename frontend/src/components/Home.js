import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Main.css';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="container">
      <div className="card">
        <h1 style={{ marginBottom: '30px' }}>🎥 Video Content Moderation System</h1>
        <button className="upload-button" onClick={() => navigate('/audio')}>🎧 Audio Moderation</button>
        <button className="upload-button" onClick={() => navigate('/video')}>🎬 Video Moderation</button>
        <button className="upload-button" onClick={() => navigate('/audiovideo')}>🔊🎥 Audio & Video Moderation</button>
      </div>
    </div>
  );
};

export default Home;
