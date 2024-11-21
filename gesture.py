import cv2
import mediapipe as mp
import time
import requests
import threading

ESP32_IP = "192.168.157.135"

# Initialize mediapipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

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

def count_fingers(hand_landmarks):
    fingers = [0, 0, 0, 0, 0]  # Thumb, index, middle, ring, pinky

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

def detect_gesture():
    last_command = None
    last_command_time = time.time()
    gesture_interval = 2  # Set a delay of 2 seconds between gesture detections

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

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

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_gesture()
