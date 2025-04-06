import os
from typing import Set, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask Configuration
FLASK_SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
FLASK_PORT = int(os.environ.get('PORT', 5000))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
FLASK_ENV = os.environ.get('FLASK_ENV', 'production')

# API Configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'
API_RATE_LIMIT = "200 per day, 50 per hour"

# File Upload Configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size
ALLOWED_EXTENSIONS: Set[str] = {
    'jpg', 'jpeg', 'png', 'webp',  # Images
    'mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv',  # Videos
    'wav', 'mp3', 'ogg', 'm4a'  # Audio
}

# Model Configuration
MODEL_PATH = 'models/model.pth'
IMG_SIZE = (224, 224)  # Increased for better accuracy
MODEL_CONFIDENCE_THRESHOLD = 0.7  # Increased threshold
MODEL_CACHE_SIZE = 1000
MODEL_UPDATE_INTERVAL = 3600  # 1 hour in seconds

# Speech Recognition Configuration
SUPPORTED_VIDEO_FORMATS: Set[str] = {'mov', 'mp4', 'avi', 'mkv', 'wmv', 'flv'}
SUPPORTED_AUDIO_FORMATS: Set[str] = {'wav', 'mp3', 'ogg', 'm4a'}
SPEECH_RECOGNITION_LANGUAGE = 'en-US'
WORD_LIST_URL = 'https://www.mit.edu/~ecprice/wordlist.10000'
SPEECH_RECOGNITION_TIMEOUT = 10  # seconds
SPEECH_RECOGNITION_RETRIES = 3

# Logging Configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/app.log'
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Data Storage
DATA_FOLDER = 'data'
PIECES_FOLDER = 'data/pieces'
SOBER_FOLDER = 'data/sober'
DRUNK_FOLDER = 'data/drunk'
MODELS_FOLDER = 'models'
LOGS_FOLDER = 'logs'
CACHE_FOLDER = 'cache'

# Database Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
DATABASE_POOL_SIZE = 5
DATABASE_MAX_OVERFLOW = 10

# Security Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key')
JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

# Feature Flags
ENABLE_EMAIL_NOTIFICATIONS = os.environ.get('ENABLE_EMAIL_NOTIFICATIONS', 'False').lower() == 'true'
ENABLE_SMS_NOTIFICATIONS = os.environ.get('ENABLE_SMS_NOTIFICATIONS', 'False').lower() == 'true'
ENABLE_API_DOCUMENTATION = os.environ.get('ENABLE_API_DOCUMENTATION', 'True').lower() == 'true'

# Notification Settings
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
SMS_API_KEY = os.environ.get('SMS_API_KEY', '')

# Ensure required directories exist
for folder in [
    UPLOAD_FOLDER, DATA_FOLDER, PIECES_FOLDER, SOBER_FOLDER, DRUNK_FOLDER,
    MODELS_FOLDER, LOGS_FOLDER, CACHE_FOLDER
]:
    os.makedirs(folder, exist_ok=True) 