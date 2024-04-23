import { useEffect, useRef } from "react";
import { Streamlit, RenderData } from "streamlit-component-lib";

const CameraCapture = ({ width, height }) => {
  const videoRef = useRef();
  const canvasRef = useRef();

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        videoRef.current.srcObject = stream;
      })
      .catch(function (error) {
        console.error("Error accessing camera:", error);
      });
  }, []);

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);
    const imageData = canvas.toDataURL("image/jpeg");

    Streamlit.setComponentValue(imageData);
  };

  return (
    <div>
      <video ref={videoRef} width={width} height={height} autoPlay></video>
      <button onClick={captureImage}>Capture Image</button>
      <canvas
        ref={canvasRef}
        style={{ display: "none" }}
        width={width}
        height={height}
      ></canvas>
    </div>
  );
};

Streamlit.componentReady();