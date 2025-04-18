import React, { useState } from "react";

const VideoPlayer = () => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [fileName, setFileName] = useState("");

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoSrc(URL.createObjectURL(file));
      setFileName(file.name);
    }
  };

  return (
    <div className="p-6 max-w-xl mx-auto bg-white rounded-2xl shadow-md space-y-4">
      <h2 className="text-2xl font-bold text-center">🎥 Video Player & Downloader</h2>

      <input
        type="file"
        accept="video/*"
        onChange={handleVideoUpload}
        className="w-full p-2 border rounded-md"
      />

      {videoSrc && (
        <>
          <video
            controls
            src={videoSrc}
            className="w-full rounded-xl mt-4"
          >
            Your browser does not support the video tag.
          </video>

          <a
            href={videoSrc}
            download={fileName}
            className="inline-block mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            ⬇️ Download Video
          </a>
        </>
      )}
    </div>
  );
};

export default VideoPlayer;
