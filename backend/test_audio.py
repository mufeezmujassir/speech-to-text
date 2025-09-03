import numpy as np
import soundfile as sf
from model_loader import ModelInference
import sys

def test_audio_processing():
    print("Testing audio processing...")
    
    # Create a simple test audio (1 second of sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print(f"Generated test audio: shape={test_audio.shape}, dtype={test_audio.dtype}")
    
    try:
        # Initialize model
        model_inference = ModelInference("models/best_asr_model.keras")
        
        # Test preprocessing
        print("Testing preprocessing...")
        features = model_inference.preprocess_audio(test_audio, sample_rate)
        print(f"Preprocessed features shape: {features.shape}")
        
        # Test transcription
        print("Testing transcription...")
        result = model_inference.transcribe(test_audio, sample_rate)
        print(f"Transcription result: '{result}'")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def test_file_audio(filepath):
    print(f"Testing audio file: {filepath}")
    
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(filepath)
        print(f"Loaded audio: shape={audio_data.shape}, dtype={audio_data.dtype}, sr={sample_rate}")
        
        # Initialize model
        model_inference = ModelInference("models/best_asr_model.keras")
        
        # Test transcription
        result = model_inference.transcribe(audio_data, sample_rate)
        print(f"Transcription result: '{result}'")
        
    except Exception as e:
        print(f"Error during file test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided audio file
        test_file_audio(sys.argv[1])
    else:
        # Test with synthetic audio
        test_audio_processing()