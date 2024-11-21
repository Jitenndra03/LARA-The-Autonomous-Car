import cv2
import mediapipe as mp
import time
import requests
import threading
import numpy as np
import tensorflow as tf
import os

# ESP32 Configuration
ESP32_IP = "192.168.157.135"

# Initialize Mediapipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load OpenCV's DNN face detector
face_detector = cv2.dnn.readNetFromCaffe(
    'models/deploy.prototxt.txt',
    'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# Webcam configuration
cap = cv2.VideoCapture(0)

# Create a "captured" folder if it doesn't exist
if not os.path.exists('captured'):
    os.makedirs('captured')

# Preprocess input image for MobileNetV2 model
def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
    image_expanded = np.expand_dims(image_preprocessed, axis=0)
    return image_expanded

# Perform face detection using OpenCV's DNN module
def detect_faces(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_boxes.append((startX, startY, endX, endY))
    return face_boxes

# Threaded function to send commands
def send_command(command):
    def task():
        try:
            url = f"http://{ESP32_IP}/{command}"
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Executed '{command}': {response.text}")
            else:
                print(f"Failed to execute '{command}'. HTTP Status: {response.status_code}")
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
    threading.Thread(target=task, daemon=True).start()

# Count raised fingers for gesture control
def count_fingers(hand_landmarks):
    fingers = [0, 0, 0, 0, 0]
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        fingers[0] = 1
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
        fingers[1] = 1
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers[2] = 1
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        fingers[3] = 1
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        fingers[4] = 1
    return sum(fingers)

def main():
    last_command = None
    last_command_time = time.time()
    gesture_interval = 2  # Delay between gesture commands
    last_capture_time = time.time()
    capture_interval = 5  # Interval for saving face captures
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Gesture detection
        command = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks)
                current_time = time.time()

                if current_time - last_command_time > gesture_interval:
                    if finger_count == 5 and last_command != "forward":
                        command = "forward"
                    elif finger_count == 2 and last_command != "backward":
                        command = "backward"
                    elif finger_count == 3 and last_command != "left":
                        command = "left"
                    elif finger_count == 4 and last_command != "right":
                        command = "right"
                    elif finger_count == 1 and last_command != "stop":
                        command = "stop"

                    if command:
                        send_command(command)
                        last_command = command
                        last_command_time = current_time

        if command:
            cv2.putText(frame, f'Command: {command}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Face detection and capture
        faces = detect_faces(frame)
        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Capture faces periodically
        if time.time() - last_capture_time > capture_interval and faces:
            cv2.imwrite(f"captured/captured_frame_{image_count}.jpg", frame)
            image_count += 1
            last_capture_time = time.time()

        cv2.imshow("Integrated Gesture & Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
