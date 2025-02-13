from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import librosa.display

import matplotlib
import matplotlib.pyplot as plt


from typing import Dict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-voice', methods=["POST"])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    # Get the audio file from the request
    audio_file = request.files['audio']

    # Save the audio file temporarily for analysis
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    y, sr = librosa.load(filepath, mono=True)

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

    matplotlib.use('Agg')

    # Plot the pitch contour
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.plot(librosa.times_like(f0), f0, label='f0 (pitch)', color='r')
    plt.title('Pitch (f0)')

    plot_path = os.path.join('static', 'plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return jsonify({'plot_url': '/static/plot.png'})


if __name__ == '__main__':
    app.run(debug=True)