import os
from typing import Set

# Flask Configuration
FLASK_SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
FLASK_PORT = int(os.environ.get('PORT', 5000))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# File Upload Configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS: Set[str] = {'jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv'}

# Model Configuration
MODEL_PATH = 'model.pth'
IMG_SIZE = (50, 50)
MODEL_CONFIDENCE_THRESHOLD = 0.6

# Speech Recognition Configuration
SUPPORTED_VIDEO_FORMATS: Set[str] = {'mov', 'mp4', 'avi', 'mkv', 'wmv', 'flv'}
SUPPORTED_AUDIO_FORMATS: Set[str] = {'wav', 'mp3', 'ogg', 'm4a'}
SPEECH_RECOGNITION_LANGUAGE = 'en-US'
WORD_LIST_URL = 'https://www.mit.edu/~ecprice/wordlist.10000'

# Logging Configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'app.log'

# Data Storage
DATA_FOLDER = 'data'
PIECES_FOLDER = 'pieces'
SOBER_FOLDER = 'sober'
DRUNK_FOLDER = 'drunk'

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, DATA_FOLDER, PIECES_FOLDER, SOBER_FOLDER, DRUNK_FOLDER]:
    os.makedirs(folder, exist_ok=True) 