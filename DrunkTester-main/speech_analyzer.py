import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import config
import soundfile as sf
from scipy import signal
import noisereduce as nr

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

class SpeechAnalyzer:
    def __init__(self):
        self.model = self._build_model()
        self.feature_cache = {}
        self.analysis_history = []
        self.sample_rate = 16000
        self.frame_length = 2048
        self.hop_length = 512
        
    def _build_model(self) -> tf.keras.Model:
        """Build a custom model for speech analysis."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(128, 128, 1)),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # LSTM layers for temporal analysis
            layers.Reshape((16, 128)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_audio(self, audio_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Preprocess audio file for analysis."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Noise reduction
            audio = nr.reduce_noise(y=audio, sr=sr)
            
            # Extract features
            features = self._extract_audio_features(audio, sr)
            
            # Create spectrogram
            spectrogram = self._create_spectrogram(audio)
            
            return spectrogram, features
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            return None
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract various audio features."""
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate': float(np.mean(zcr)),
                'pitch_mean': float(np.mean(pitches[magnitudes > np.max(magnitudes)/2])),
                'duration': len(audio) / sr
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return {}
    
    def _create_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Create mel spectrogram from audio."""
        try:
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mels=128
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Resize to fixed size
            mel_spec_db = cv2.resize(mel_spec_db, (128, 128))
            
            # Add channel dimension
            mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
            
            return mel_spec_db
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {str(e)}")
            return np.zeros((1, 128, 128, 1))
    
    def analyze_speech(self, audio_path: str) -> Dict:
        """Analyze speech for signs of intoxication."""
        try:
            # Check cache
            audio_hash = self._get_audio_hash(audio_path)
            if audio_hash in self.feature_cache:
                return self.feature_cache[audio_hash]
            
            # Preprocess audio
            processed_data = self._preprocess_audio(audio_path)
            if processed_data is None:
                raise ValueError("Failed to preprocess audio")
            
            spectrogram, features = processed_data
            
            # Make prediction
            prediction = self.model.predict(spectrogram)
            confidence = float(np.max(prediction))
            is_intoxicated = bool(np.argmax(prediction))
            
            # Create analysis result
            result = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'is_intoxicated': is_intoxicated,
                'features': features,
                'raw_prediction': prediction.tolist()
            }
            
            # Update cache and history
            self.feature_cache[audio_hash] = result
            self.analysis_history.append(result)
            
            # Save to history file
            self._save_analysis_history()
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing speech: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_audio_hash(self, audio_path: str) -> str:
        """Generate hash for audio file."""
        with open(audio_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _save_analysis_history(self):
        """Save analysis history to file."""
        try:
            history_file = Path(config.DATA_FOLDER) / 'speech_analysis_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analysis history: {str(e)}")
    
    def get_analysis_history(self) -> List[Dict]:
        """Get analysis history."""
        return self.analysis_history
    
    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                   validation_data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 10, batch_size: int = 32):
        """Train the model on new data."""
        try:
            self.model.fit(
                train_data,
                train_labels,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise 