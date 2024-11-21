import cv2
import mediapipe as mp

# Initialize mediapipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(2)

def count_fingers(hand_landmarks):
    # List to store the state of each finger (1 = up, 0 = down)
    fingers = [0, 0, 0, 0, 0]  # Thumb, index, middle, ring, pinky

    # Thumb - check if the tip is above the base
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        fingers[0] = 1
    
    # Index Finger - check if tip is above the base
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
        fingers[1] = 1
    
    # Middle Finger - check if tip is above the base
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers[2] = 1
    
    # Ring Finger - check if tip is above the base
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        fingers[3] = 1
    
    # Pinky Finger - check if tip is above the base
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        fingers[4] = 1
    
    # Count the number of fingers that are up
    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for a mirror effect (Optional)
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of fingers
            finger_count = count_fingers(hand_landmarks)

            # Control the car based on the finger count
            if finger_count == 5:
                gesture = "Move Forward"
                # Send forward signal to the car
            elif finger_count == 2:
                gesture = "Move Backward"
                # Send backward signal to the car
            elif finger_count == 3:
                gesture = "Turn Left"
                # Send left turn signal to the car
            elif finger_count == 4:
                gesture = "Turn Right"
                # Send right turn signal to the car
            elif finger_count == 1:
                gesture = "Stop"
                # Send stop signal to the car
            else:
                gesture = "No Gesture"

            # Show gesture on the screen
            cv2.putText(frame, f'Fingers: {finger_count} ({gesture})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the gesture information
    cv2.imshow("Hand Gesture Control", frame)

    # Exit condition on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
