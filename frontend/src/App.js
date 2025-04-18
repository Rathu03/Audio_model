import AudioUploader from "./components/AudioUploader";
import AudioVideo from "./components/AudioVideo";
import Home from "./components/Home";
import VideoUploader from "./components/VideoUploader";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import "./styles/Main.css";
import VideoPlayer from "./components/VideoPlayer";

const App = () => {
    return (
      <div style={{ display: "flex", justifyContent: "space-between", gap: "80px" }}>
          <Router>
            <Routes>
              <Route path="/" element={<Home />}/>
              <Route path="/audio" element={<AudioUploader />}/>
              <Route path="/video" element={<VideoUploader />}/>
              <Route path="/audiovideo" element={<AudioVideo />}/>
              <Route path="/temp" element={<VideoPlayer />}/>
            </Routes>
          </Router>  
      </div>
    );
}

export default App;
