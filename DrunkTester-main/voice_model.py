import requests
import random
import speech_recognition as sr
from pydub import AudioSegment
import logging
from typing import Optional, Tuple
import os
import config
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SUPPORTED_VIDEO_FORMATS = {'mov', 'mp4', 'avi', 'mkv', 'wmv', 'flv'}
SUPPORTED_AUDIO_FORMATS = {'wav', 'mp3', 'ogg', 'm4a'}

# Check if someones voice is slurring

# Get phrase to say
@lru_cache(maxsize=100)
def get_phrase(num: int = 3) -> str:
    """Get a random phrase from a word list with caching."""
    try:
        response = requests.get(config.WORD_LIST_URL, timeout=5)
        response.raise_for_status()
        
        all_words = [word.decode('utf-8') for word in response.content.splitlines()]
        phrase = ' '.join(random.choice(all_words) for _ in range(num))
        return phrase.strip()
    except Exception as e:
        logger.error(f"Error getting phrase: {str(e)}")
        return "energetic happiness"  # Fallback phrase

# Convert speech to text
def speech_to_text(file_path: str, retries: int = 3) -> Optional[str]:
    """Convert speech to text using Google's Speech Recognition with retries."""
    recognizer = sr.Recognizer()
    
    for attempt in range(retries):
        try:
            with sr.AudioFile(file_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_text = recognizer.record(source)
                
                # Try different recognition services
                try:
                    return recognizer.recognize_google(audio_text, language=config.SPEECH_RECOGNITION_LANGUAGE)
                except sr.UnknownValueError:
                    logger.warning(f"Speech recognition attempt {attempt + 1} failed: Could not understand audio")
                except sr.RequestError as e:
                    logger.warning(f"Speech recognition attempt {attempt + 1} failed: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in speech_to_text attempt {attempt + 1}: {str(e)}")
            
    logger.error("All speech recognition attempts failed")
    return None


# Implementation of the levenshtein distance algorythm to see the distance (insertions, deletions) between strings
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    m, n = len(s1) + 1, len(s2) + 1
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[m - 1][n - 1]

# I want a percentile though, so I'll create a function to do that
def similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings as a percentage."""
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)

def get_file_hash(file_path: str) -> str:
    """Generate a hash for the audio file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=100)
def cached_check_slurring(file_hash: str, example: str) -> float:
    """Cached version of check_slurring using file hash."""
    try:
        text = speech_to_text(file_hash)
        if text is None:
            return 1.0  # Return maximum slurring score if speech recognition fails
            
        similar_percent = similarity(text.lower(), example.lower())
        return 1 - similar_percent
    except Exception as e:
        logger.error(f"Error in cached_check_slurring: {str(e)}")
        return 1.0  # Return maximum slurring score on error

def check_slurring(file_path: str, example: str) -> float:
    """Check if speech is slurred by comparing with example phrase."""
    try:
        file_hash = get_file_hash(file_path)
        return cached_check_slurring(file_hash, example)
    except Exception as e:
        logger.error(f"Error in check_slurring: {str(e)}")
        return 1.0  # Return maximum slurring score on error

def get_wav_file(video_path: str, output_path: str = 'test.wav') -> bool:
    """Convert video to WAV format with improved error handling."""
    try:
        # Get file extension
        ext = os.path.splitext(video_path)[1][1:].lower()
        
        if ext not in config.SUPPORTED_VIDEO_FORMATS:
            logger.error(f"Unsupported video format: {ext}")
            return False
            
        # Try to convert the video to audio
        try:
            audio = AudioSegment.from_file(video_path, format=ext)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Export as WAV
            audio.export(output_path, format='wav')
            return True
            
        except Exception as e:
            logger.error(f"Error converting video to WAV: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in get_wav_file: {str(e)}")
        return False

def get_words() -> str:
    """Read the target phrase from file with error handling."""
    try:
        with open('phrase.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error reading phrase.txt: {str(e)}")
        return "energetic happiness"  # Fallback phrase

if __name__ =='main':
    toxic, toxicity = check_slurring('hi.wav', 'hello hello hello hello hello hello hello hello')
