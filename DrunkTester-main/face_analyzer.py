import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import dlib
import logging
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import config

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

class FaceAnalyzer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.model = self._build_model()
        self.feature_cache = {}
        self.analysis_history = []
        
    def _build_model(self) -> tf.keras.Model:
        """Build a custom CNN model for intoxication detection."""
        model = models.Sequential([
            # Feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Feature analysis
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _extract_facial_features(self, image: np.ndarray) -> Dict:
        """Extract detailed facial features using dlib."""
        try:
            faces = self.detector(image)
            if not faces:
                return {}
            
            face = faces[0]
            landmarks = self.predictor(image, face)
            
            features = {
                'face_position': {
                    'x': face.left(),
                    'y': face.top(),
                    'width': face.width(),
                    'height': face.height()
                },
                'landmarks': {
                    'eyes': self._get_eye_points(landmarks),
                    'mouth': self._get_mouth_points(landmarks),
                    'jaw': self._get_jaw_points(landmarks)
                },
                'measurements': {
                    'eye_aspect_ratio': self._calculate_eye_aspect_ratio(landmarks),
                    'mouth_aspect_ratio': self._calculate_mouth_aspect_ratio(landmarks),
                    'jaw_angle': self._calculate_jaw_angle(landmarks)
                }
            }
            
            return features
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return {}
    
    def _get_eye_points(self, landmarks) -> List[Tuple[int, int]]:
        """Extract eye landmark points."""
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 48)]
    
    def _get_mouth_points(self, landmarks) -> List[Tuple[int, int]]:
        """Extract mouth landmark points."""
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
    
    def _get_jaw_points(self, landmarks) -> List[Tuple[int, int]]:
        """Extract jaw landmark points."""
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
    
    def _calculate_eye_aspect_ratio(self, landmarks) -> float:
        """Calculate eye aspect ratio for drowsiness detection."""
        # Implementation of eye aspect ratio calculation
        return 0.0
    
    def _calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """Calculate mouth aspect ratio for speech analysis."""
        # Implementation of mouth aspect ratio calculation
        return 0.0
    
    def _calculate_jaw_angle(self, landmarks) -> float:
        """Calculate jaw angle for head position analysis."""
        # Implementation of jaw angle calculation
        return 0.0
    
    def _preprocess_image(self, image_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Preprocess image for model input."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract facial features
            features = self._extract_facial_features(image)
            
            # Resize and normalize
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image, features
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def analyze_face(self, image_path: str) -> Dict:
        """Analyze face for signs of intoxication."""
        try:
            # Check cache
            image_hash = self._get_image_hash(image_path)
            if image_hash in self.feature_cache:
                return self.feature_cache[image_hash]
            
            # Preprocess image
            processed_data = self._preprocess_image(image_path)
            if processed_data is None:
                raise ValueError("Failed to preprocess image")
            
            image, features = processed_data
            
            # Make prediction
            prediction = self.model.predict(image)
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
            self.feature_cache[image_hash] = result
            self.analysis_history.append(result)
            
            # Save to history file
            self._save_analysis_history()
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _save_analysis_history(self):
        """Save analysis history to file."""
        try:
            history_file = Path(config.DATA_FOLDER) / 'analysis_history.json'
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