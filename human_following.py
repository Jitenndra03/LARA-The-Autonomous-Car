import cv2
import requests
from itertools import cycle

# ESP32 configuration
ESP32_IP = "192.168.157.135"  # Replace with your ESP32's IP address

def send_command(command):
    try:
        url = f"http://{ESP32_IP}/{command}"
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Executed '{command}': {response.text}")
        else:
            print(f"Failed to execute '{command}'. HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"Error sending command '{command}': {e}")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # Take the first detected face
        x, y, w, h = faces[0]
        cx = x + w // 2  # Center x-coordinate of the face
        return cx, frame.shape[1]
    return None, frame.shape[1]

# Alternating directions for searching
search_directions = cycle(["slight_left", "slight_right"])

def move_car_based_on_position(cx, frame_width):
    if cx is None:
        send_command("stop")
    elif cx < frame_width // 3:
        send_command("right")
    elif cx > 2 * frame_width // 3:
        send_command("left")
    else:
        send_command("forward")

# Main Control Loop
cap = cv2.VideoCapture(1)  # Use your camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection
    cx, frame_width = detect_face(frame)
    move_car_based_on_position(cx, frame_width)
    print(f"Face Center X: {cx}")

    # Show the video with detections (optional)
    if cx:
        cv2.putText(frame, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, frame.shape[0] // 2), 5, (0, 255, 0), -1)
    cv2.imshow("Haar Cascade Face Following", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
