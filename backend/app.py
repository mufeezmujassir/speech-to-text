from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
import io
import numpy as np
import os
from model_loader import ModelInference

app = Flask(__name__)
CORS(app)

# Initialize model
model_inference = None

def load_model():
    global model_inference
    model_path = "models/best_asr_model.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure to copy your trained model to the models/ directory")
        return False
    
    try:
        model_inference = ModelInference(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    if model_inference is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500
    return jsonify({"status": "healthy"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    global model_inference
    
    if model_inference is None:
        return jsonify({
            "error": "Model not loaded",
            "success": False
        }), 500
    
    try:
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided",
                "success": False
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                "error": "No audio file selected",
                "success": False
            }), 400
        
        # Read audio data
        audio_bytes = io.BytesIO(audio_file.read())
        
        try:
            audio_data, sample_rate = sf.read(audio_bytes)
        except Exception as e:
            return jsonify({
                "error": f"Invalid audio file format: {str(e)}",
                "success": False
            }), 400
        
        # Validate audio
        if len(audio_data) == 0:
            return jsonify({
                "error": "Empty audio file",
                "success": False
            }), 400
        
        # Debug info
        print(f"Audio info - Shape: {audio_data.shape}, Dtype: {audio_data.dtype}, Sample rate: {sample_rate}")
        
        # Convert to proper numpy array if needed
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Ensure float32 dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Limit audio length (e.g., max 30 seconds)
        max_samples = 30 * sample_rate
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
            print(f"Audio trimmed to 30 seconds")
        
        print(f"Processing audio: {len(audio_data)} samples at {sample_rate}Hz")
        print(f"Final audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        
        # Transcribe
        transcription = model_inference.transcribe(audio_data, sample_rate)
        
        print(f"Transcription result: {transcription}")
        
        return jsonify({
            "transcription": transcription,
            "success": True,
            "audio_duration": len(audio_data) / sample_rate,
            "sample_rate": sample_rate
        })
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({
            "error": f"Transcription failed: {str(e)}",
            "success": False
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the server is working"""
    return jsonify({
        "message": "Speech-to-Text API is running",
        "model_loaded": model_inference is not None
    })

if __name__ == '__main__':
    print("Starting Speech-to-Text API server...")
    print("Loading model...")
    
    if load_model():
        print("Server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Server not started.")
        print("Please check:")
        print("1. Model file exists at models/best_asr_model.keras")
        print("2. All custom classes are properly defined in model_loader.py")
        print("3. Dependencies are installed correctly")