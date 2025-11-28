"""
Service de Prédiction d'Images - Classification avec Deep Learning
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model


class ImagePredictionService:
    """Service pour la classification d'images avec TensorFlow"""
    
    # Chemins par défaut
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_MODEL_PATH = BASE_DIR / 'ml' / 'imageclassifier.h5'
    
    # Configuration
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    IMAGE_SIZE = (256, 256)
    
    # Instance singleton
    _instance = None
    _model = None
    _is_loaded = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le service et charge le modèle"""
        if not ImagePredictionService._is_loaded:
            print("loading image model ...")
            self.load_model()
    
    @classmethod
    def load_model(cls:'ImagePredictionService', model_path=None):
        """
        Charge le modèle TensorFlow
        
        Args:
            model_path (str): Chemin vers le modèle .h5
            
        Returns:
            bool: True si chargement réussi
        """
        if model_path:
            model_path = Path(model_path)
        else:
            model_path = Path(os.environ.get('IMAGE_MODEL_PATH', cls.DEFAULT_MODEL_PATH))
        
        print(f"🔍 Chargement du modèle d'images:")
        print(f"   📄 Modèle: {model_path.absolute()}")
        
        try:
            
            if model_path.exists():
                print(model_path)
                cls._model = load_model(str(model_path))
                cls._is_loaded = True
                print(f"✅ Modèle d'images chargé avec succès!")
                print(f"   Architecture: {cls._model.summary()}")
                return True
            else:
                print(f"❌ Modèle non trouvé: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            cls._is_loaded = False
            return False
    
    @classmethod
    def is_model_loaded(cls:'ImagePredictionService'):
        """Vérifie si le modèle est chargé"""
        return cls._is_loaded and cls._model is not None
    
    @classmethod
    def get_model_info(cls:'ImagePredictionService'):
        """Retourne les informations sur le modèle"""
        if not cls._is_loaded:
            return {
                'loaded': False,
                'message': 'Modèle non chargé',
                'path': str(cls.DEFAULT_MODEL_PATH.absolute()),
                'exists': cls.DEFAULT_MODEL_PATH.exists()
            }
        
        return {
            'loaded': True,
            'model_type': type(cls._model).__name__,
            'input_shape': str(cls._model.input_shape),
            'output_shape': str(cls._model.output_shape),
            'path': str(cls.DEFAULT_MODEL_PATH.absolute())
        }
    
    @staticmethod
    def allowed_file(filename):
        """Vérifie si l'extension du fichier est autorisée"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ImagePredictionService.ALLOWED_EXTENSIONS
    
    @classmethod
    def preprocess_image(cls:'ImagePredictionService', image_path):
        """
        Prétraite une image pour la prédiction
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            np.ndarray: Image prétraitée
        """
        try:
            # Lire l'image
            img = cv2.imread(str(image_path))
            
            if img is None:
                raise ValueError(f"Impossible de lire l'image: {image_path}")
            
            # Convertir BGR (OpenCV) en RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionner
            img_resized = tf.image.resize(img, cls.IMAGE_SIZE)
            
            # Normaliser (0-1)
            img_normalized = img_resized / 255.0
            
            # Ajouter dimension batch
            img_batch = np.expand_dims(img_normalized, 0)
            
            return img_batch
            
        except Exception as e:
            raise ValueError(f"Erreur lors du prétraitement: {str(e)}")
    
    @classmethod
    def preprocess_image_from_bytes(cls:'ImagePredictionService', image_bytes):
        """
        Prétraite une image depuis des bytes
        
        Args:
            image_bytes (bytes): Données de l'image
            
        Returns:
            np.ndarray: Image prétraitée
        """
        try:
            # Convertir bytes en array numpy
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Décoder l'image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Impossible de décoder l'image")
            
            # Convertir BGR en RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionner
            img_resized = tf.image.resize(img, cls.IMAGE_SIZE)
            
            # Normaliser
            img_normalized = img_resized / 255.0
            
            # Ajouter dimension batch
            img_batch = np.expand_dims(img_normalized, 0)
            
            return img_batch
            
        except Exception as e:
            raise ValueError(f"Erreur lors du prétraitement: {str(e)}")
    
    @classmethod
    def predict(cls:'ImagePredictionService', image_input, threshold=0.5):
        """
        Prédit la classe d'une image
        
        Args:
            image_input: Chemin de l'image ou bytes
            threshold (float): Seuil de décision (0-1)
            
        Returns:
            dict: Résultat de la prédiction
        """
        if not cls.is_model_loaded():
            return {
                'success': False,
                'error': 'Modèle non chargé',
                'prediction': None
            }
        
        try:
            # Prétraiter l'image
            if isinstance(image_input, (str, Path)):
                img_processed = cls.preprocess_image(image_input)
            elif isinstance(image_input, bytes):
                img_processed = cls.preprocess_image_from_bytes(image_input)
            else:
                raise ValueError("Type d'entrée non supporté")
            
            # Faire la prédiction
            prediction = cls._model.predict(img_processed, verbose=0)[0][0]
            
            # Interpréter le résultat
            # Votre modèle: >0.5 = Sad, <=0.5 = Happy
            is_sad = float(prediction) > threshold
            confidence = float(prediction) if is_sad else 1 - float(prediction)
            
            return {
                'success': True,
                'prediction': float(prediction),
                'class': 'Sad' if is_sad else 'Happy',
                'confidence': round(confidence * 100, 2),
                'threshold': threshold,
                'probabilities': {
                    'sad': round(float(prediction) * 100, 2),
                    'happy': round((1 - float(prediction)) * 100, 2)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur lors de la prédiction: {str(e)}',
                'prediction': None
            }
    
    @classmethod
    def predict_batch(cls:'ImagePredictionService', image_paths, threshold=0.5):
        """
        Prédit pour plusieurs images
        
        Args:
            image_paths (list): Liste de chemins d'images
            threshold (float): Seuil de décision
            
        Returns:
            dict: Résultats des prédictions
        """
        if not cls.is_model_loaded():
            return {
                'success': False,
                'error': 'Modèle non chargé',
                'predictions': []
            }
        
        try:
            results = []
            sad_count = 0
            
            for idx, image_path in enumerate(image_paths):
                result = cls.predict(image_path, threshold)
                result['index'] = idx
                result['image'] = str(image_path)
                results.append(result)
                
                if result.get('success') and result.get('class') == 'Sad':
                    sad_count += 1
            
            return {
                'success': True,
                'total': len(image_paths),
                'sad_count': sad_count,
                'happy_count': len(image_paths) - sad_count,
                'predictions': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur batch: {str(e)}',
                'predictions': []
            }