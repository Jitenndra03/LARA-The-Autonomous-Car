import os
import time
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import pygame

# Initialize audio playback
pygame.mixer.init()


# Function to play audio responses
def play_response(command):
    audio_files = {
        "move forward": "Audio_files/forward.wav",
        "move move forward": "Audio_files/forward.wav",
        "forward": "Audio_files/forward.wav",
        "stop": "Audio_files/stop.wav",
        "turn left": "Audio_files/left.wav",
        "left": "Audio_files/left.wav",
        "move left": "Audio_files/left.wav",
        "turn right": "Audio_files/right.wav",
        "right": "Audio_files/right.wav",
        "move right": "Audio_files/right.wav",
        "move backward": "Audio_files/backward.wav",
        "move move backward": "Audio_files/backward.wav",
        "backward": "Audio_files/backward.wav",
        "hello": "Audio_files/hello.wav",
        "who are you": "Audio_files/whoru.wav"
    }
    file_path = audio_files.get(command)
    if file_path and os.path.exists(file_path):
        try:
            sound = pygame.mixer.Sound(file_path)
            sound.play()
        except pygame.error as e:
            print(f"Audio playback error: {e}")
    else:
        print(f"No audio response found for command: {command}")


# Initialize Vosk model
model_path = "models/vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Download it from https://alphacephei.com/vosk/models.")
    exit(1)

model = Model(model_path)

# Define grammar as an array of strings
grammar = [
    "move forward", "move move forward", "forward",
    "stop",
    "turn left", "left", "move left",
    "turn right", "right", "move right",
    "move backward", "backward", "move move backward",
    "hello",
    "who are you"
]

# Initialize recognizer
recognizer = KaldiRecognizer(model, 16000)
recognizer.SetWords(True)  # Optional: enables word-level recognition
recognizer.SetGrammar(json.dumps(grammar))  # Apply the grammar as an array of strings

# Audio input setup
p = pyaudio.PyAudio()

# Try opening the audio stream
try:
    stream = p.open(format=pyaudio.paInt16, channels=8, rate=16000, input=True, frames_per_buffer=4000,
                    input_device_index=2)
    stream.start_stream()
except OSError as e:
    print(f"Error initializing audio stream: {e}")
    p.terminate()
    exit(1)

print("Listening for commands...")

# Main loop
try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            command = result.get('text', '')

            if command:  # If a command is recognized
                print(f"Recognized command: {command}")

                # Play the corresponding response
                play_response(command)

                # Inject motor driver code here
                if command in grammar:
                    print(f"Executing motor driver code for {command}...")
                    # Add your motor driver control code here
                elif command == "stop":
                    print("Executing motor driver code to stop...")
                    # Add your motor driver control code here

                # Add a delay to avoid listening to random commands
                print("Waiting for 2 seconds before resuming...")
                time.sleep(0.5)  # Delay to prevent misrecognition of ambient noise

        else:
            # Handle partial results or do nothing
            pass

except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Cleanup resources
    print("Cleaning up resources...")
    if stream.is_active():
        stream.stop_stream()
        stream.close()
    p.terminate()
    pygame.mixer.quit()
    print("Resources cleaned up.")
