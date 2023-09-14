from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import soundfile
import librosa
from pydub import AudioSegment
from io import BytesIO  # Importa BytesIO

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de Scikit-Learn
model = joblib.load("speach_emotion_model.joblib")

# Función para extraer características del archivo de audio
def extract_feature(file):
    with soundfile.SoundFile(file) as sound_file:  # Usa el objeto file directamente
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        result = np.array([])

        # Características MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

        # Características Chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))

        # Características Mel
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

    return result

@app.post("/predict/")
async def predict_emotion(audio_file: UploadFile = File(...)):
    try:
        # Convertir el archivo de audio a formato .wav
        audio_data = await audio_file.read()  # Lee el contenido del archivo
        try:
            audio = AudioSegment.from_file(BytesIO(audio_data), format="webm")
        except:
            audio = AudioSegment.from_file(BytesIO(audio_data), format="mp4")
        
        # Convertir audio a formato WAV con una frecuencia de muestreo consistente
        audio = audio.set_channels(1).set_frame_rate(16000)  # Establece 1 canal y 16 kHz de frecuencia de muestreo
        audio_file_converted_to_wav = audio.export(format="wav", codec="pcm_s16le")

        # Realizar la extracción de características
        extracted_features = extract_feature(audio_file_converted_to_wav)

        # Realizar la predicción
        pred = model.predict(extracted_features.reshape(1, -1))

        return {"emotional_prediction": str(pred[0])}
    except Exception as e:
        return {"error": str(e)}
