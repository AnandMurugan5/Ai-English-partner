import speech_recognition as sr
import pyttsx3
from model import model_speech

def audio_files(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_and_process(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        print("Reading audio file...")
        audio = recognizer.record(source)

        try:
            print("Recognizing text...")
            recognized_data = recognizer.recognize_whisper(
                audio,
                model="medium.en",
                show_dict=True
            )
            text = recognized_data.get("text", "")

            if text.lower().strip() == "stop":
                print("Received stop command, exiting...")
                return text, "Stopped"

            response_text = model_speech(text)
            return text, response_text

        except sr.UnknownValueError:
            return "Sorry, I didn't catch that. Could you please repeat?", ""
        except sr.RequestError as e:
            return f"Could not request results; {e}", ""
        except Exception as e:
            return f"An error occurred: {e}", ""
