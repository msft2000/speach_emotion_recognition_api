from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import soundfile
import librosa

app = FastAPI()

# Cargar el modelo de Scikit-Learn
model = joblib.load("speach_emotion_model.joblib")

# Función para extraer características del archivo de audio
def extract_feature(file):
    with soundfile.SoundFile(file.file) as sound_file:  # Usar 'file.file' para acceso al contenido binario
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
        # Realizar la extracción de características
        extracted_features = extract_feature(audio_file)

        #Realice prediction
        pred = model.predict(extracted_features.reshape(1, -1))
        
        return {"emotional_prediction": str(pred[0])}
    except Exception as e:
        return {"error": str(e)}

