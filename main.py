"""
Application Flask - API de détection de fraude bancaire
Architecture modulaire avec séparation des responsabilités
"""
from flask import Flask, jsonify
import os

# Imports locaux
from db.connexion.connexion import init_db, create_tables, test_connection
from config.jwt_config import init_jwt
from routes.user_routes import user_bp
from routes.transaction_routes import transaction_bp
from routes.prediction_routes import prediction_bp
from routes.auth_routes import auth_bp

# Configuration du port
port = int(os.environ.get("PORT", 8080))

# Initialisation de Flask
app = Flask(__name__)

# Configuration Flask
app.config['WTF_CSRF_ENABLED'] = False  # Désactivé pour API REST
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ============================================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================================

# Initialiser la connexion à la base de données
db = init_db(app)

# ============================================================================
# INITIALISATION JWT
# ============================================================================

jwt = init_jwt(app)

# ============================================================================
# CRÉATION DES TABLES
# ============================================================================

# Créer les tables au démarrage (après JWT pour avoir UserAuth)
create_tables(app)

# ============================================================================
# ENREGISTREMENT DES BLUEPRINTS (ROUTES)
# ============================================================================

app.register_blueprint(auth_bp)  # Routes d'authentification
app.register_blueprint(user_bp)
app.register_blueprint(transaction_bp)
app.register_blueprint(prediction_bp)

# ============================================================================
# ROUTES PRINCIPALES
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil - Documentation API"""
    return jsonify({
        'message': 'API de détection de fraude bancaire',
        'version': '2.0.0',
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
    
    # Récupération du chemin de la base de données SQLite
    db_path = os.environ.get("DATABASE_PATH", "fraud_detection.db")
    if not os.path.isabs(db_path):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, db_path)
    
    # Vérification de l'existence du fichier
    db_exists = os.path.exists(db_path)
    db_size = os.path.getsize(db_path) if db_exists else 0
    
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
        'api': {
            'version': '2.0.0',
            'environment': os.environ.get('ENV', 'production')
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
    print("🚀 DÉMARRAGE DE L'API - Détection de fraude bancaire")
    print("="*80)
    print(f"📍 Port: {port}")
    print(f"🗄️  Base de données: {os.environ.get('MYSQL_DATABASE', 'fraud_detection')}")
    print(f"🔗 Host: {os.environ.get('MYSQL_HOST', 'localhost')}")
    print("="*80 + "\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get('ENV', 'production') == 'local'
    )