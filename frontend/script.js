class SpeechToTextApp {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        this.recordBtn = document.getElementById('recordBtn');
        this.status = document.getElementById('status');
        this.timer = document.getElementById('timer');
        this.audioFileInput = document.getElementById('audioFile');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.transcriptionBox = document.getElementById('transcription');
        this.loading = document.getElementById('loading');
    }
    
    bindEvents() {
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.uploadBtn.addEventListener('click', () => this.uploadFile());
    }
    
    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.transcribeAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.startTime = Date.now();
            
            this.recordBtn.textContent = 'Stop Recording';
            this.recordBtn.classList.add('recording');
            this.status.textContent = 'Recording...';
            
            this.startTimer();
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showError('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.recordBtn.textContent = 'Start Recording';
            this.recordBtn.classList.remove('recording');
            this.status.textContent = 'Processing...';
            
            this.stopTimer();
        }
    }
    
    startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            this.timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        this.timer.textContent = '00:00';
    }
    
    async uploadFile() {
        const file = this.audioFileInput.files[0];
        if (!file) {
            this.showError('Please select an audio file');
            return;
        }
        
        await this.transcribeAudio(file);
    }
    
    async transcribeAudio(audioBlob) {
        this.showLoading(true);
        this.status.textContent = 'Transcribing...';
        
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob);
            
            const response = await fetch('http://localhost:5000/transcribe', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayTranscription(result.transcription);
                this.status.textContent = 'Transcription complete';
            } else {
                this.showError(result.error || 'Transcription failed');
            }
            
        } catch (error) {
            console.error('Error during transcription:', error);
            this.showError('Failed to connect to server');
        } finally {
            this.showLoading(false);
        }
    }
    
    displayTranscription(text) {
        this.transcriptionBox.textContent = text || 'No transcription available';
        this.transcriptionBox.classList.remove('error');
    }
    
    showError(message) {
        this.transcriptionBox.textContent = message;
        this.transcriptionBox.classList.add('error');
        this.status.textContent = 'Error occurred';
    }
    
    showLoading(show) {
        this.loading.classList.toggle('hidden', !show);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SpeechToTextApp();
});