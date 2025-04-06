# DrunkTester 2.0 - Advanced Intoxication Detection System

A sophisticated machine learning-based system for detecting potential intoxication through facial analysis and speech pattern recognition. This system uses advanced deep learning models and signal processing techniques to provide accurate and reliable intoxication detection.

## Features

- **Advanced Facial Analysis**
  - Real-time face detection using dlib
  - Facial landmark detection (68 points)
  - Eye aspect ratio calculation for drowsiness detection
  - Mouth aspect ratio analysis for speech pattern recognition
  - Jaw angle measurement for head position analysis
  - Deep learning-based intoxication classification

- **Speech Pattern Analysis**
  - Mel spectrogram generation for audio analysis
  - MFCC feature extraction
  - Spectral feature analysis
  - Pitch detection and analysis
  - Noise reduction and audio enhancement
  - LSTM-based temporal pattern recognition

- **Combined Analysis**
  - Integration of facial and speech analysis
  - Weighted confidence scoring
  - Comprehensive intoxication assessment

- **Security & Performance**
  - JWT-based authentication
  - Rate limiting for API protection
  - Feature caching for improved performance
  - Analysis history tracking
  - Secure file handling

## Technical Architecture

### Face Analysis Module
- TensorFlow-based CNN architecture
- Dlib for facial landmark detection
- Haar cascades for face detection
- Feature extraction and analysis pipeline
- Caching system for improved performance

### Speech Analysis Module
- TensorFlow-based hybrid CNN-LSTM model
- Librosa for audio processing
- Mel spectrogram generation
- Advanced feature extraction
- Temporal pattern recognition

### API Layer
- Flask REST API with Swagger documentation
- JWT authentication
- Rate limiting
- Error handling and logging
- History tracking and caching

## Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Dlib
- OpenCV
- Librosa
- Flask and related packages
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaryapatel09/DrunkTester2.0.git
cd DrunkTester2.0
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# Download dlib shape predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

5. Configure the application:
- Copy `config.example.py` to `config.py`
- Update configuration settings as needed

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the API documentation:
- Open your browser and navigate to `http://localhost:5000/docs`

3. API Endpoints:
- `/api/v1/login` - Authentication
- `/api/v1/analyze` - Perform intoxication analysis
- `/api/v1/history` - View analysis history
- `/api/v1/clear-cache` - Clear feature cache
- `/api/v1/train` - Train models on new data

4. Example API Usage:
```python
import requests

# Login
response = requests.post('http://localhost:5000/api/v1/login', json={
    'username': 'admin',
    'password': 'admin'
})
token = response.json()['access_token']

# Analyze face
response = requests.post('http://localhost:5000/api/v1/analyze', 
    headers={'Authorization': f'Bearer {token}'},
    json={
        'file_path': 'path/to/image.jpg',
        'analysis_type': 'face'
    }
)
```

## Configuration

The application can be configured through environment variables or by modifying `config.py`:

- `FLASK_ENV` - Application environment (development/production)
- `FLASK_DEBUG` - Debug mode
- `FLASK_PORT` - Port number
- `JWT_SECRET_KEY` - JWT secret key
- `API_RATE_LIMIT` - API rate limit
- `MODEL_PATH` - Path to model files
- `UPLOAD_FOLDER` - Upload directory
- `DATA_FOLDER` - Data storage directory
- `LOGS_FOLDER` - Log directory
- `CACHE_FOLDER` - Cache directory

## Model Training

The system supports model training with new data:

1. Prepare training data:
- Organize images in `data/train/sober` and `data/train/drunk`
- Organize audio files in `data/train/sober_audio` and `data/train/drunk_audio`

2. Start training:
```bash
curl -X POST http://localhost:5000/api/v1/train \
  -H "Authorization: Bearer $TOKEN"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Dlib developers for facial landmark detection
- Librosa team for audio processing
- Flask team for the web framework
