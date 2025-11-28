"""
Application Flask - API de détection de fraude bancaire
Architecture modulaire avec séparation des responsabilités
"""
from flask import Flask, jsonify
import os

# Imports locaux
from config.cors_config import init_cors
from db.connexion.connexion import init_db, create_tables, test_connection
from config.jwt_config import init_jwt
from routes.user_routes import user_bp
from routes.transaction_routes import transaction_bp
from routes.prediction_routes import prediction_bp
from routes.images_routes import image_bp
from routes.auth_routes import auth_bp
from services.prediction_service import PredictionService
from services.image_prediction_service import ImagePredictionService

# Configuration du port
port = int(os.environ.get("PORT", 8080))

# Initialisation de Flask
app = Flask(__name__)

# Configuration Flask
app.config['WTF_CSRF_ENABLED'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# ============================================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================================

db = init_db(app)

# ============================================================================
# INITIALISATION JWT
# ============================================================================

jwt = init_jwt(app)

# ============================================================================
# CRÉATION DES TABLES
# ============================================================================

create_tables(app)

# ============================================================================
# CONFIGURATION CORS
# ============================================================================

init_cors(app)

# ============================================================================
# ENREGISTREMENT DES BLUEPRINTS (ROUTES)
# ============================================================================

app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(transaction_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(image_bp)  # Nouveau blueprint pour les images

# ============================================================================
# ROUTES PRINCIPALES
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil - Documentation API"""
    return jsonify({
        'message': 'API de détection de fraude bancaire',
        'version': '2.1.0',
        'status': 'online',
        'documentation': {
            'health': {
                'GET /health': 'Vérifier l\'état de l\'API et de la base de données'
            },
            'users': {
                'GET /users': 'Liste des utilisateurs (pagination)',
                'GET /users/<id>': 'Détails d\'un utilisateur',
                'POST /users': 'Créer un utilisateur',
                'PUT /users/<id>': 'Mettre à jour un utilisateur',
                'DELETE /users/<id>': 'Supprimer un utilisateur',
                'GET /users/search?q=': 'Rechercher des utilisateurs',
                'GET /users/stats': 'Statistiques des utilisateurs'
            },
            'predictions': {
                'GET /predict/status': 'Statut du modèle ML',
                'POST /predict/reload': 'Recharger le modèle',
                'POST /predict/transaction': 'Prédire une transaction',
                'POST /predict/transaction/explain': 'Analyser les facteurs de risque',
                'POST /predict/batch': 'Prédire un batch de transactions',
                'POST /predict/transaction/<id>': 'Prédire pour une transaction en BDD',
                'POST /predict/transactions/pending': 'Prédire toutes les transactions en attente'
            },
            'image_predictions': {
                'GET /predict/image/status': 'Statut du modèle d\'images',
                'POST /predict/image/reload': 'Recharger le modèle d\'images',
                'POST /predict/image/predict': 'Prédire une image (Happy/Sad)',
                'POST /predict/image/predict/file': 'Prédire depuis un chemin',
                'POST /predict/image/predict/batch': 'Prédire plusieurs images'
            },
            'transactions': {
                'GET /transactions': 'Liste des transactions (pagination)',
                'GET /transactions/<id>': 'Détails d\'une transaction',
                'POST /transactions': 'Créer une transaction',
                'PUT /transactions/<id>': 'Mettre à jour une transaction',
                'DELETE /transactions/<id>': 'Supprimer une transaction',
                'GET /transactions/fraud': 'Transactions frauduleuses',
                'GET /transactions/account/<account_no>': 'Transactions d\'un compte',
                'GET /transactions/stats': 'Statistiques des transactions',
                'GET /transactions/search': 'Rechercher par montant',
                'POST /transactions/<id>/mark-fraud': 'Marquer comme fraude'
            }
        }
    })


@app.route('/health')
def health_check():
    """Vérification de santé de l'API et de la connexion DB"""
    db_test = test_connection()
    
    db_path = os.environ.get("DATABASE_PATH", "fraud_detection.db")
    if not os.path.isabs(db_path):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, db_path)
    
    db_exists = os.path.exists(db_path)
    db_size = os.path.getsize(db_path) if db_exists else 0
    
    # Vérifier les modèles
    fraud_model_loaded = PredictionService.is_model_loaded()
    image_model_loaded = ImagePredictionService.is_model_loaded()
    
    return jsonify({
        'status': 'healthy' if db_test['status'] == 'success' else 'unhealthy',
        'database': {
            'type': 'SQLite',
            'status': db_test['status'],
            'message': db_test['message'],
            'config': {
                'path': db_path,
                'exists': db_exists,
                'size_bytes': db_size,
                'size_mb': round(db_size / (1024 * 1024), 2) if db_exists else 0
            }
        },
        'models': {
            'fraud_detection': {
                'loaded': fraud_model_loaded,
                'status': 'ready' if fraud_model_loaded else 'not loaded'
            },
            'image_classification': {
                'loaded': image_model_loaded,
                'status': 'ready' if image_model_loaded else 'not loaded'
            }
        },
        'api': {
            'version': '2.1.0',
            'environment': os.environ.get('ENV', 'production')
        }
    })


@app.route('/debug/model-status')
def model_status():
    """Diagnostic complet des modèles"""
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parent
    ml_dir = base_dir / 'ml'
    
    return jsonify({
        'working_directory': str(Path.cwd()),
        'base_directory': str(base_dir),
        'ml_directory_exists': ml_dir.exists(),
        'ml_files': list(str(f) for f in ml_dir.glob('*.*')) if ml_dir.exists() else [],
        'fraud_model': PredictionService.get_model_info(),
        'image_model': ImagePredictionService.get_model_info(),
        'environment': {
            'MODEL_PATH': os.environ.get('MODEL_PATH', 'Not set'),
            'SCALER_PATH': os.environ.get('SCALER_PATH', 'Not set'),
            'STATS_PATH': os.environ.get('STATS_PATH', 'Not set'),
            'IMAGE_MODEL_PATH': os.environ.get('IMAGE_MODEL_PATH', 'Not set'),
        }
    })


# ============================================================================
# GESTION DES ERREURS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Erreur 404 - Route non trouvée"""
    return jsonify({
        'success': False,
        'error': 'Route non trouvée',
        'code': 404
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Erreur 405 - Méthode non autorisée"""
    return jsonify({
        'success': False,
        'error': 'Méthode HTTP non autorisée pour cette route',
        'code': 405
    }), 405


@app.errorhandler(413)
def file_too_large(error):
    """Erreur 413 - Fichier trop volumineux"""
    return jsonify({
        'success': False,
        'error': 'Fichier trop volumineux (max: 16 MB)',
        'code': 413
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Erreur 500 - Erreur serveur interne"""
    return jsonify({
        'success': False,
        'error': 'Erreur serveur interne',
        'code': 500
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    """Gestion globale des exceptions"""
    return jsonify({
        'success': False,
        'error': str(error),
        'type': type(error).__name__
    }), 500


# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DE L'API - Détection de fraude bancaire + Classification d'images")
    print("="*80)
    print(f"📍 Port: {port}")
    print(f"🗄️  Base de données: {os.environ.get('DATABASE_PATH', 'fraud_detection.db')}")
    print(f"🤖 Modèle fraude: {PredictionService.is_model_loaded()}")
    print(f"🖼️  Modèle image: {ImagePredictionService.is_model_loaded()}")
    print("="*80 + "\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get('ENV', 'production') == 'local'
    )