import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import logging
from typing import Optional, Tuple, Dict
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

class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.convolution(x)

@lru_cache(maxsize=100)
def load_model(model_path: str = config.MODEL_PATH) -> Optional[ImprovedModel]:
    """Load the trained model from disk with caching."""
    try:
        model = ImprovedModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    """Preprocess the input image for model prediction."""
    try:
        img = Image.open(image_path)
        img_transformed = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])(img)
        return img_transformed.unsqueeze(0)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def get_image_hash(image_path: str) -> str:
    """Generate a hash for the image file."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=100)
def cached_check_intoxicated(image_hash: str) -> Tuple[float, bool]:
    """Cached version of check_intoxicated using image hash."""
    try:
        model = load_model()
        if model is None:
            raise RuntimeError("Failed to load model")

        img_batch = preprocess_image(image_hash)
        if img_batch is None:
            raise RuntimeError("Failed to preprocess image")

        with torch.no_grad():
            output = model(img_batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            return confidence.item(), bool(predicted_class.item())

    except Exception as e:
        logger.error(f"Error in cached_check_intoxicated: {str(e)}")
        raise

def check_intoxicated(image_path: str) -> Tuple[float, bool]:
    """
    Check if a person in the image appears intoxicated.
    Returns a tuple of (confidence_score, is_intoxicated)
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
