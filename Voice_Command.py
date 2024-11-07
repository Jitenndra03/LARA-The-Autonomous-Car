# Import necessary libraries
import speech_recognition as sr
import time
import Jetson.GPIO as GPIO

# Setup GPIO
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering

# Define GPIO pins for motor control
# Adjust these pin numbers according to your wiring
LEFT_MOTOR_PIN = 11
RIGHT_MOTOR_PIN = 13
FORWARD_MOTOR_PIN = 15
BACKWARD_MOTOR_PIN = 16

# Set up motor pins as output
GPIO.setup(LEFT_MOTOR_PIN, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_PIN, GPIO.OUT)
GPIO.setup(FORWARD_MOTOR_PIN, GPIO.OUT)
GPIO.setup(BACKWARD_MOTOR_PIN, GPIO.OUT)

# Define robot movement functions
def move_left():
    print("Moving Left")
    GPIO.output(LEFT_MOTOR_PIN, GPIO.HIGH)
    time.sleep(1)  # Duration for moving left
    GPIO.output(LEFT_MOTOR_PIN, GPIO.LOW)

def move_right():
    print("Moving Right")
    GPIO.output(RIGHT_MOTOR_PIN, GPIO.HIGH)
    time.sleep(1)  # Duration for moving right
    GPIO.output(RIGHT_MOTOR_PIN, GPIO.LOW)

def move_forward():
    print("Moving Forward")
    GPIO.output(FORWARD_MOTOR_PIN, GPIO.HIGH)
    time.sleep(1)  # Duration for moving forward
    GPIO.output(FORWARD_MOTOR_PIN, GPIO.LOW)

def move_backward():
    print("Moving Backward")
    GPIO.output(BACKWARD_MOTOR_PIN, GPIO.HIGH)
    time.sleep(1)  # Duration for moving backward
    GPIO.output(BACKWARD_MOTOR_PIN, GPIO.LOW)

def stop():
    print("Stopping")
    GPIO.output(LEFT_MOTOR_PIN, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_PIN, GPIO.LOW)
    GPIO.output(FORWARD_MOTOR_PIN, GPIO.LOW)
    GPIO.output(BACKWARD_MOTOR_PIN, GPIO.LOW)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Main function to capture voice commands and control the robot
def listen_and_move():
    with sr.Microphone() as source:
        print("Listening for commands...")
        
        while True:
            # Capture audio from the microphone
            audio_data = recognizer.listen(source)
            try:
                # Recognize the speech and convert it to text
                command = recognizer.recognize_google(audio_data).lower()
                print(f"Command received: {command}")
                
                # Check for specific commands and call the respective function
                if "left" in command:
                    move_left()
                elif "right" in command:
                    move_right()
                elif "forward" in command:
                    move_forward()
                elif "backward" in command:
                    move_backward()
                elif "stop" in command:
                    stop()
                    print("Exiting command loop.")
                    break  # Exit the loop on "stop"
                else:
                    print("Unknown command. Please say left, right, forward, backward, or stop.")
                
                # Pause briefly to prevent overlapping commands
                time.sleep(1)
            
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError:
                print("Request to Google Speech Recognition API failed")

# Clean up GPIO settings on exit
def cleanup():
    GPIO.cleanup()

# Run the function to start listening and controlling the robot
if __name__ == "__main__":
    try:
        listen_and_move()
    finally:
        cleanup()
