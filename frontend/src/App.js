import AudioUploader from "./components/AudioUploader";
import VideoUploader from "./components/VideoUploader";
import "./styles/Main.css";

const App = () => {
    return (
      <div style={{ display: "flex", justifyContent: "space-between", gap: "80px" }}>
        <AudioUploader />
        <VideoUploader />
    </div>
    );
}

export default App;
