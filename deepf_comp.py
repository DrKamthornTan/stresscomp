import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from deepface import DeepFace

st.set_page_config(page_title='DHV Stress Demo', layout='centered')

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = ["stressless", "stressful"]

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

import os
import tempfile

def predict_camera():
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to 224x224 and normalize the pixel values
    resized_frame = cv2.resize(rgb_frame, (224, 224))
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1

    # Load the frame into the array
    data[0] = normalized_frame

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the frame and prediction
    st.image(rgb_frame, channels="RGB", caption=f"{class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}")
    
    caption = f"<h1 style='color: {'blue' if class_name == 'stressless' else 'red'}'>Class: {class_name} ({class_name.replace('stress', '')})</h1>\n<h1>Confidence Score: {confidence_score:.2f}</h1>"
    st.markdown(caption, unsafe_allow_html=True)

    if class_name == "stressless":
        st.markdown("<h1 style='font-size: 24px;'>You've done good so far.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 24px;'>Take a deep breath! Sit back and relax. Try exercise, vacation, entertainment, or consult your physician.</h1>", unsafe_allow_html=True)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    # Save the PIL Image as a temporary file in the default temporary directory
    _, temp_path = tempfile.mkstemp(suffix=".jpg", dir=tempfile.gettempdir())
    pil_image.save(temp_path)

    # Perform stress detection using DeepFace
    results = DeepFace.analyze(img_path=temp_path, actions=['age'])  # Analyze image using DeepFace
    predicted_age = results[0]['age']
    st.markdown(f"<h1>Predicted Age: {predicted_age}</h1>", unsafe_allow_html=True)

# Set the title and subtitle
st.title("DHV AI Startup Stress Detection Demo")
st.subheader("การประเมินความเครียดจากสีหน้าโดยกล้องหรือภาพถ่าย")
st.write("โปรดอยู่ห่างกล้องประมาณ 2 ฟุต มองกล้อง อยู่นิ่งและกดปุ่ม Start Camera")

# Create a button to start the camera
if st.button("Start Camera"):
    # Predict using the camera
    predict_camera()

import streamlit as st

# Define the width and height for the camera capture component
capture_width = 640
capture_height = 480

# Display the camera capture component
st.components.v1.html(
    f"""
    <div>
        <h2></h2>
        <p></p>
        <div id="cameraCapture"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/react@17.0.2/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@17.0.2/umd/react-dom.production.min.js"></script>
    <script>
        {open('camera_capture.js', 'r').read()}
        ReactDOM.render(
            React.createElement(CameraCapture, {{
                width: {capture_width},
                height: {capture_height}
            }}),
            document.getElementById("cameraCapture")
        );
    </script>
    """,
    height=600,
)

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image = Image.open(image_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image and prediction
    caption = f"<h1 style='font-size: 24px; color: {'blue' if class_name == 'stressless' else 'red'}'>Class: {class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}</h1>"
    st.image(image, caption=None)
    st.markdown(caption, unsafe_allow_html=True)
    if class_name == "stressless":
        st.markdown("<h1 style='font-size: 24px;'>You've done good so far.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 24px;'>Take a deep breath! Sit back and relax. Try exercise, vacation, entertainment, or consult your physician.</h1>", unsafe_allow_html=True)

    # Perform stress detection using DeepFace
    results = DeepFace.analyze(image_file, actions=['age'])
    # Iterate over the list and find the dictionary with 'age' information
    predicted_age = None

    if results:
    # Assume the first element in the list has the 'age' information
        first_result = results[0]

        # Check if the first_result is a dictionary and contains the 'age' key
        if isinstance(first_result, dict) and 'age' in first_result:
            predicted_age = first_result['age']

   
    st.markdown(f"<h1>Predicted Age: {predicted_age}</h1>", unsafe_allow_html=True)
    



