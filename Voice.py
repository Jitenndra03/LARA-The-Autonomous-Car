import json
import os
import pyaudio
from vosk import Model, KaldiRecognizer
from gtts import gTTS
import tempfile
# pip install mpg321 before running the programm 

# Initialize Vosk model for speech recognition
model = Model("/home/jitendra/Desktop/Project Cheeta/LARA-The-Autonomous-Car/models/vosk-model-small-en-us-0.15")
commands_grammar = '["move forward", "move backward", "turn left", "turn right", "stop", "hello"]'
recognizer = KaldiRecognizer(model, 16000, commands_grammar)

# Set up PyAudio for capturing audio from the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=12000, input=True, frames_per_buffer=2000)
stream.start_stream()

# Function to speak the response in real time using gTTS
def speak(response_text):
    tts = gTTS(text=response_text, lang='en')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_file_name = temp_audio.name
        # Save the TTS audio to a temporary file
        tts.save(temp_file_name)
    # Play the audio file using aplay for low latency
    os.system(f"mpg321 -q {temp_file_name}")  # Ensure mpg321 is installed for mp3 playback
    os.remove(temp_file_name)  # Delete the file after playback

# Function to handle recognized commands and generate a response
def handle_command(command):
    if command == "move forward":
        response = "Car is moving forward."
        # Add code to move the car forward
    elif command == "move backward":
        response = "Car is moving backward."
        # Add code to move the car backward
    elif command == "turn left":
        response = "Car is turning left."
        # Add code to turn the car left
    elif command == "turn right":
        response = "Car is turning right."
        # Add code to turn the car right
    elif command == "stop":
        response = "Car has stopped Quiting the program"
        speak(response)
        return 
        # Add code to stop the car
    elif command == "hello":
        response = "Hello, user!"
    else:
        response = "Command not recognized."

    print(response)  # Print for debugging
    speak(response)  # Speak the response

print("Listening for commands...")

# Start the recognition loop
try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        
        if recognizer.AcceptWaveform(data):
            result_json = recognizer.Result()
            result = json.loads(result_json)
            command = result.get("text", "")
            
            if command:
                print(f"Recognized command: {command}")
                handle_command(command)
except KeyboardInterrupt:
    print("\nStopping voice recognition...")

# Clean up resources
stream.stop_stream()
stream.close()
audio.terminate()