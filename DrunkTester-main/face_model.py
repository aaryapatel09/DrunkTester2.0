import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
from typing import Optional, Tuple, Dict, List
import config
from functools import lru_cache
import hashlib
import cv2
import numpy as np
from datetime import datetime

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

class AdvancedFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )
        
        # Feature analysis layers
        self.feature_analyzer = nn.Sequential(
            nn.Linear(512 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Get classification
        classification = self.classifier(features)
        
        # Get feature analysis
        feature_analysis = self.feature_analyzer(features)
        
        return classification, feature_analysis

def preprocess_image(image_path: str) -> Optional[Tuple[torch.Tensor, List[Dict]]]:
    """Preprocess the input image and extract facial features."""
    try:
        # Load and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Process each face
        processed_faces = []
        for (x, y, w, h) in faces:
            # Extract face
            face = img[y:y+h, x:x+w]
            
            # Resize face
            face = cv2.resize(face, config.IMG_SIZE)
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            face_tensor = transform(Image.fromarray(face)).unsqueeze(0)
            
            # Extract facial landmarks
            landmarks = extract_facial_landmarks(face)
            
            processed_faces.append({
                'tensor': face_tensor,
                'landmarks': landmarks,
                'position': {'x': x, 'y': y, 'w': w, 'h': h}
            })
        
        return processed_faces[0]['tensor'], processed_faces
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def extract_facial_landmarks(face: np.ndarray) -> Dict:
    """Extract facial landmarks using dlib."""
    try:
        # This is a placeholder for actual landmark detection
        # In a real implementation, you would use dlib or another library
        return {
            'eyes': [],
            'nose': [],
            'mouth': [],
            'jaw': []
        }
    except Exception as e:
        logger.error(f"Error extracting facial landmarks: {str(e)}")
        return {}

@lru_cache(maxsize=config.MODEL_CACHE_SIZE)
def load_model(model_path: str = config.MODEL_PATH) -> Optional[AdvancedFaceModel]:
    """Load the trained model from disk with caching."""
    try:
        model = AdvancedFaceModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def get_image_hash(image_path: str) -> str:
    """Generate a hash for the image file."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=config.MODEL_CACHE_SIZE)
def cached_check_intoxicated(image_hash: str) -> Tuple[float, bool, Dict]:
    """Cached version of check_intoxicated using image hash."""
    try:
        model = load_model()
        if model is None:
            raise RuntimeError("Failed to load model")

        processed_data = preprocess_image(image_hash)
        if processed_data is None:
            raise RuntimeError("Failed to preprocess image")
            
        face_tensor, face_data = processed_data

        with torch.no_grad():
            classification, feature_analysis = model(face_tensor)
            probabilities = torch.softmax(classification, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Extract features
            features = {
                'landmarks': face_data[0]['landmarks'],
                'position': face_data[0]['position'],
                'feature_vector': feature_analysis.numpy().tolist()
            }
            
            return confidence.item(), bool(predicted_class.item()), features

    except Exception as e:
        logger.error(f"Error in cached_check_intoxicated: {str(e)}")
        raise

def check_intoxicated(image_path: str) -> Tuple[float, bool, Dict]:
    """
    Check if a person in the image appears intoxicated.
    Returns a tuple of (confidence_score, is_intoxicated, features)
    """
    try:
        image_hash = get_image_hash(image_path)
        return cached_check_intoxicated(image_hash)
    except Exception as e:
        logger.error(f"Error in check_intoxicated: {str(e)}")
        raise

def organize_training_data():
    """Organize training data into sober and drunk folders."""
    try:
        for filename in os.listdir(config.PIECES_FOLDER):
            if filename.endswith('.jpg'):
                number = int(filename.replace('piece', '').replace('.jpg', ''))
                target_folder = config.SOBER_FOLDER if number % 2 == 0 else config.DRUNK_FOLDER
                source_path = os.path.join(config.PIECES_FOLDER, filename)
                target_path = os.path.join(target_folder, filename)
                
                if not os.path.exists(target_path):
                    os.rename(source_path, target_path)
    except Exception as e:
        logger.error(f"Error organizing training data: {str(e)}")
        raise

# Note: Training code has been removed as it should be in a separate training script
