"""
Routes pour les prédictions d'images
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
from services.image_prediction_service import ImagePredictionService
import os
from pathlib import Path

# Créer un Blueprint
image_bp = Blueprint('images', __name__, url_prefix='/predict/image')

# Configuration upload
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

# Initialiser le service de prédiction
prediction_service = ImagePredictionService()


@image_bp.route('/status', methods=['GET'])
@jwt_required()
def model_status():
    """Vérifier le statut du modèle d'images"""
    try:
        info = prediction_service.get_model_info()
        
        return jsonify({
            'success': True,
            'model': info
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_bp.route('/reload', methods=['POST'])
@jwt_required()
def reload_model():
    """Recharger le modèle depuis le fichier"""
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path')
        
        success = prediction_service.load_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Modèle rechargé avec succès',
                'model': prediction_service.get_model_info()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Échec du rechargement'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict_image():
    """
    Prédire la classe d'une image (Happy/Sad)
    
    Multipart form-data:
    - image: fichier image (PNG, JPG, JPEG, BMP)
    - threshold: seuil optionnel (défaut: 0.5)
    """
    try:
        # Vérifier qu'un fichier est présent
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Aucun fichier fourni (clé: "image")'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Aucun fichier sélectionné'
            }), 400
        
        # Vérifier l'extension
        if not prediction_service.allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Extension non autorisée. Utilisez: {", ".join(prediction_service.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Vérifier la taille
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'Fichier trop volumineux (max: {MAX_FILE_SIZE / (1024*1024)} MB)'
            }), 400
        
        # Récupérer le seuil optionnel
        threshold = float(request.form.get('threshold', 0.5))
        if not 0 <= threshold <= 1:
            return jsonify({
                'success': False,
                'error': 'Le seuil doit être entre 0 et 1'
            }), 400
        
        # Lire les bytes de l'image
        image_bytes = file.read()
        
        # Faire la prédiction
        result = prediction_service.predict(image_bytes, threshold)
        
        if not result['success']:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'filename': secure_filename(file.filename),
            'file_size': file_size,
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_bp.route('/predict/file', methods=['POST'])
@jwt_required()
def predict_image_from_path():
    """
    Prédire depuis un chemin de fichier
    
    Body JSON:
    {
        "image_path": "/path/to/image.jpg",
        "threshold": 0.5  // optionnel
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Chemin de l\'image requis (clé: "image_path")'
            }), 400
        
        image_path = data['image_path']
        threshold = float(data.get('threshold', 0.5))
        
        if not 0 <= threshold <= 1:
            return jsonify({
                'success': False,
                'error': 'Le seuil doit être entre 0 et 1'
            }), 400
        
        # Vérifier que le fichier existe
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'error': f'Fichier non trouvé: {image_path}'
            }), 404
        
        # Faire la prédiction
        result = prediction_service.predict(image_path, threshold)
        
        if not result['success']:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'image_path': image_path,
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_bp.route('/predict/batch', methods=['POST'])
@jwt_required()
def predict_batch():
    """
    Prédire pour plusieurs images
    
    Multipart form-data:
    - images: plusieurs fichiers images
    - threshold: seuil optionnel
    """
    try:
        # Vérifier qu'il y a des fichiers
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Aucun fichier fourni (clé: "images")'
            }), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'Liste de fichiers vide'
            }), 400
        
        if len(files) > 50:
            return jsonify({
                'success': False,
                'error': 'Maximum 50 images par batch'
            }), 400
        
        threshold = float(request.form.get('threshold', 0.5))
        
        results = []
        sad_count = 0
        
        for idx, file in enumerate(files):
            if file.filename == '':
                continue
            
            if not prediction_service.allowed_file(file.filename):
                results.append({
                    'index': idx,
                    'filename': file.filename,
                    'success': False,
                    'error': 'Extension non autorisée'
                })
                continue
            
            try:
                image_bytes = file.read()
                result = prediction_service.predict(image_bytes, threshold)
                result['index'] = idx
                result['filename'] = secure_filename(file.filename)
                results.append(result)
                
                if result.get('success') and result.get('class') == 'Sad':
                    sad_count += 1
                    
            except Exception as e:
                results.append({
                    'index': idx,
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'sad_count': sad_count,
            'happy_count': len(files) - sad_count,
            'predictions': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500