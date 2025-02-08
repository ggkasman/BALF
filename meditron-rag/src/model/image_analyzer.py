"""
Image analysis module using ResNet50 for leukocyte ratio detection.
"""
import logging
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
from PIL import Image

logger = logging.getLogger(__name__)

class ResNet50Model:
    """ResNet50 model for leukocyte classification."""
    
    def __init__(self):
        # Create base model
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        # Freeze base model
        self.base_model.trainable = False
        
        # Create new model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = self.base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
    def __call__(self, x):
        """Forward pass (similar to PyTorch's forward)."""
        return self.model(x)

class LeukocyteAnalyzer:
    """ResNet50 model for leukocyte ratio analysis."""
    
    def __init__(self, model_path: str):
        """Initialize the analyzer.
        
        Args:
            model_path: Path to the saved ResNet50 model
        """
        try:
            # Convert to absolute path and resolve any symlinks
            model_path = str(Path(model_path).resolve())
            logger.info(f"Loading model from absolute path: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Set up GPU configuration (similar to PyTorch's device setup)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Set GPU memory growth")
                    self.device = "GPU"
                except RuntimeError as e:
                    logger.warning(f"Error setting GPU memory growth: {e}")
                    self.device = "CPU"
            else:
                self.device = "CPU"
            logger.info(f"Using device: {self.device}")
            
            # Initialize model (PyTorch-like structure)
            try:
                logger.info("Attempting to load model...")
                self.model = ResNet50Model()
                self.model.model.load_weights(model_path)
                logger.info(f"Successfully loaded model from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
            
            # Define image transformations (similar to PyTorch transforms)
            self.input_size = (224, 224)
            logger.info(f"Set input size to: {self.input_size}")
            
            # Define class names
            self.class_names = ['neutrophils', 'eosinophils', 'lymphocytes', 'macrophages']
            logger.info(f"Initialized class names: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LeukocyteAnalyzer: {str(e)}")
            raise
            
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        try:
            logger.info(f"Preprocessing image: {image_path}")
            
            # Verify image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            
            # Read and preprocess image (similar to PyTorch transforms)
            image = tf.keras.preprocessing.image.load_img(
                image_path,
                target_size=self.input_size,
                color_mode='rgb'
            )
            
            # Convert to array and normalize
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = image_array / 255.0  # Normalize to [0,1]
            image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
            
            # Apply ImageNet normalization (like PyTorch)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image_array = (image_array - mean) / std
            
            logger.info(f"Preprocessed image shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
            
    @tf.function  # Similar to PyTorch's @torch.no_grad()
    def predict(self, x):
        """Forward pass with no gradient computation."""
        return self.model(x)
            
    def analyze(self, image_path: str) -> Dict[str, float]:
        """Analyze the image and return cell type ratios.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping cell types to their ratios
        """
        try:
            # Preprocess the image
            logger.info(f"Starting analysis of image: {image_path}")
            img_array = self.preprocess_image(image_path)
            
            # Run model prediction
            logger.info("Running model prediction...")
            raw_predictions = self.predict(img_array)
            logger.info(f"Raw predictions: {raw_predictions}")
            
            # Convert predictions to dictionary
            analysis_results = dict(zip(self.class_names, raw_predictions[0].numpy()))
            logger.info(f"Analysis results: {analysis_results}")
            
            # Format results for display
            formatted_results = {
                f"{cell_type.capitalize()} Ratio": f"{ratio:.2%}"
                for cell_type, ratio in analysis_results.items()
            }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise 