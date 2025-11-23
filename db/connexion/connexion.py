"""
Configuration de la connexion à la base de données MySQL
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import QueuePool
import os

# Instance SQLAlchemy (sera initialisée dans main.py)
db = SQLAlchemy()


def init_db(app):
    """
    Initialise la connexion à la base de données
    
    Args:
        app: Instance Flask
    """
    # Récupération des variables d'environnement
    mysql_host = os.environ.get("MYSQL_HOST", "localhost")
    mysql_port = os.environ.get("MYSQL_PORT", "3306")
    mysql_user = os.environ.get("MYSQL_USER", "root")
    mysql_password = os.environ.get("MYSQL_PASSWORD", "")
    mysql_database = os.environ.get("MYSQL_DATABASE", "fraud_detection")
    
    # Construction de l'URL de connexion MySQL
    database_url = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}"
    
    # Configuration SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,              # Nombre de connexions dans le pool
        'pool_recycle': 3600,         # Recycler les connexions après 1h
        'pool_pre_ping': True,        # Vérifier la connexion avant utilisation
        'pool_timeout': 30,           # Timeout pour obtenir une connexion
        'max_overflow': 5,            # Connexions supplémentaires autorisées
        'poolclass': QueuePool,       # Type de pool
        'echo': False                 # Logs SQL (mettre True en dev)
    }
    
    # Initialiser SQLAlchemy avec l'app Flask
    db.init_app(app)
    
    print(f"🔗 Connexion à MySQL: {mysql_user}@{mysql_host}:{mysql_port}/{mysql_database}")
    
    return db


def create_tables(app):
    """
    Crée toutes les tables définies dans les modèles
    
    Args:
        app: Instance Flask
    """
    with app.app_context():
        try:
            db.create_all()
            print("✅ Tables créées avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la création des tables : {e}")


def test_connection():
    """
    Teste la connexion à la base de données
    
    Returns:
        dict: Résultat du test avec statut et message
    """
    try:
        db.session.execute(db.text('SELECT 1'))
        return {
            "status": "success",
            "message": "Connexion à la base de données réussie"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur de connexion : {str(e)}"
        }