from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import model_speech
import speech_recognition as sr
from pydub import AudioSegment
import os

app = Flask(__name__)
CORS(app)

def convert_to_pcm_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")

def listen_and_process(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            recognized_data = recognizer.recognize_google(audio)
            text = recognized_data
            response_text = model_speech(text)  # Assuming you have a model_speech function
            return text, response_text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that. Could you please repeat?", ""
        except sr.RequestError as e:
            return f"Could not request results; {e}", ""
        except Exception as e:
            return f"An error occurred: {e}", ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recording():
    return jsonify({"message": "Recording started"}), 200

@app.route('/stop', methods=['POST'])
def stop_recording():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)
    pcm_audio_path = "pcm_audio.wav"

    try:
        convert_to_pcm_wav(audio_path, pcm_audio_path)
        recognized_text, response_text = listen_and_process(pcm_audio_path)
        os.remove(audio_path)
        os.remove(pcm_audio_path)
        return jsonify({"recognized_text": recognized_text, "response_text": response_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
