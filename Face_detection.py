import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Load OpenCV's DNN face detector
face_detector = cv2.dnn.readNetFromCaffe(
    'models/deploy.prototxt.txt',  # Path to the deploy prototxt file
    'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'  # Pre-trained model file
)

# Load the MobileNetV2 model
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocess input image for MobileNetV2 model
def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))  # Resize to 224x224
    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
    image_expanded = np.expand_dims(image_preprocessed, axis=0)
    return image_expanded

# Perform face detection using OpenCV's DNN module
def detect_faces(image):
    h, w = image.shape[:2]
    # Convert image to blob for DNN face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_boxes.append((startX, startY, endX, endY))
    return face_boxes

# Ensure "captured" folder exists
if not os.path.exists('captured'):
    os.makedirs('captured')

# Start webcam video stream
video_stream = cv2.VideoCapture(0)

# Capture frames every 5 seconds
capture_interval = 2  # 5 seconds
last_capture_time = time.time()

image_count = 0

# Set a frame skip interval
frame_skip = 2  # Number of frames to skip
frame_counter = 0  # Frame counter

while True:
    ret, frame = video_stream.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break

    # Increment the frame counter
    frame_counter += 1

    # Skip processing for specified frames
    if frame_counter % frame_skip != 0:
        continue  # Skip the frame

    # Detect faces in the current frame
    faces = detect_faces(frame)

    # Draw bounding boxes around detected faces
    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Check if 5 seconds have passed since the last capture
    if time.time() - last_capture_time > capture_interval:
        if faces:  # Only save if at least one face is detected
            # Save the entire frame as an image file inside the "captured" folder
            cv2.imwrite(f"captured/captured_frame_{image_count}.jpg", frame)
            image_count += 1
            last_capture_time = time.time()  # Reset the capture timer

    # Display the resulting frame
    cv2.imshow("Webcam Face Detection", frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_stream.release()
cv2.destroyAllWindows()
