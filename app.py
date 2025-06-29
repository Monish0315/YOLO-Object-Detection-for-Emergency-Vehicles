import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import os

# Load the trained YOLO model
model = YOLO(r"C:\Users\pmoni\Traffic Management System\model.pt")

# Traffic signal logic function
def traffic_signal_logic(detected_emergency):
    """
    Updates the traffic signal states based on the presence of emergency vehicles.

    Args:
        detected_emergency (bool): Whether an emergency vehicle was detected.

    Returns:
        dict: A dictionary of signal states for all directions.
    """
    signals = {"North": "Red", "South": "Red", "East": "Red", "West": "Red"}
    
    if detected_emergency:
        # If an emergency vehicle is detected, prioritize the West signal
        signals["West"] = "Green"
    return signals

# Function to perform inference and return the image with bounding boxes
def predict(image):
    # Run inference
    results = model(image)

    # Extract the image with bounding boxes
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Extract detected object labels
    detected_labels = [box.cls for box in results[0].boxes]
    detected_names = [model.names[int(cls)] for cls in detected_labels]

    # Calculate metrics
    num_objects = len(detected_labels)
    metrics = f"Emergency Vehicles Detected: {num_objects} \n"
    if num_objects > 5:
        metrics+= "Heavy traffic detected - Check Another Route"
    else:
        metrics+= "Less traffic detected - You may proceed"

    # Check if any emergency vehicle is detected
    emergency_labels = {"FireEngineOff", "Ambulance", "PoliceCar", "siren"}  # Add your emergency vehicle labels here
    detected_emergency = any(label in emergency_labels for label in detected_names)

    # Update traffic signals based on detection
    signals = traffic_signal_logic(detected_emergency)

    return result_image, metrics, signals

# Function to load traffic signal images
def load_traffic_signal_images():
    """
    Loads images for traffic signals (Red, Yellow, Green).
    """
    signal_images = {
        "Red": Image.open("red_light.png"),  # Replace with your red light image path
        "Yellow": Image.open("yellow_light.png"),  # Replace with your yellow light image path
        "Green": Image.open("green_light.png"),  # Replace with your green light image path
    }
    return signal_images

# Streamlit app
def main():
    st.set_page_config(page_title="Emergency Vehicle Detection", layout="wide")
    st.title("🚨 Emergency Vehicle Detection and Traffic Signal Simulation 🚨")

    # Logo and instructions
    logo_path = r"C:\Users\pmoni\Traffic Management System\logo.jpg"
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        st.image(logo_image, width=100)
    st.markdown("Upload an image or capture one using your camera to detect emergency vehicles and simulate traffic signal behavior.")

    # Load traffic signal images
    signal_images = load_traffic_signal_images()

    # Option to choose between file upload and camera
    option = st.radio("Choose an option:", ("Upload Image", "Use Camera"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Capture an Image")

    # Traffic signal placeholders
    initial_signals = {"North": "Yellow", "South": "Yellow", "East": "Yellow", "West": "Yellow"}
    signal_placeholder = st.empty()

    # Display initial traffic signals
    with signal_placeholder.container():
        st.markdown("### Initial Traffic Signals (Yellow)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(signal_images["Yellow"], caption="North", width=100)
        with col2:
            st.image(signal_images["Yellow"], caption="South", width=100)
        with col3:
            st.image(signal_images["Yellow"], caption="East", width=100)
        with col4:
            st.image(signal_images["Yellow"], caption="West", width=100)

    # Add "Detect" button
    if uploaded_file is not None:
        st.markdown("### Uploaded/Captured Image")
        try:
            # Handle both file upload and camera input
            if isinstance(uploaded_file, bytes) or isinstance(uploaded_file, str):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            else:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)

            # Decode image
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Error: Unable to decode the image. Please upload a valid image file.")
            else:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, caption="Uploaded/Captured Image", use_container_width=True)

                if st.button("Detect"):
                    # Perform predictions
                    result_image, metrics, signals = predict(image)

                    # Display detected image
                    st.markdown("### Detected Vehicles")
                    st.image(result_image, channels="RGB", caption="Detected Image", use_container_width=True)

                    # Display detection metrics
                    st.markdown("### Detection Metrics")
                    st.text(metrics)

                    # Simulate yellow signal
                    time.sleep(2)
                    st.markdown("### Simulating Traffic Signals...")
                    time.sleep(3)

                    # Update final traffic signals
                    signal_placeholder.empty()
                    with signal_placeholder.container():
                        st.markdown("### Final Traffic Signals")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.image(signal_images[signals["North"]], caption="North", width=100)
                        with col2:
                            st.image(signal_images[signals["South"]], caption="South", width=100)
                        with col3:
                            st.image(signal_images[signals["East"]], caption="East", width=100)
                        with col4:
                            st.image(signal_images[signals["West"]], caption="West", width=100)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("Powered by **YOLOv11** | Created by Our Team")

if __name__ == "__main__":
    main()
