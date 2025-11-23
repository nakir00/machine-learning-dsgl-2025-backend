"""
Configuration de la connexion à la base de données SQLite
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import StaticPool
import os

# Instance SQLAlchemy (sera initialisée dans main.py)
db = SQLAlchemy()


def init_db(app):
    """
    Initialise la connexion à la base de données SQLite
    
    Args:
        app: Instance Flask
    """
    # Chemin vers le fichier SQLite
    db_path = os.environ.get("DATABASE_PATH", "db.sqlite3")
    
    # Si chemin relatif, le placer dans le dossier du projet
    if not os.path.isabs(db_path):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, db_path)
    
    # Construction de l'URL de connexion SQLite
    database_url = f"sqlite:///{db_path}"
    
    # Configuration SQLAlchemy pour SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'poolclass': StaticPool,      # Pool adapté pour SQLite
        'connect_args': {
            'check_same_thread': False,  # Permet l'accès multi-thread
            'timeout': 30                # Timeout pour les locks
        },
        'echo': False                    # Logs SQL (mettre True en dev)
    }
    
    # Initialiser SQLAlchemy avec l'app Flask
    db.init_app(app)
    
    print(f"🔗 Connexion à SQLite: {db_path}")
    
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
    Teste la connexion à la base de données SQLite
    
    Returns:
        dict: Résultat du test avec statut et message
    """
    try:
        db.session.execute(db.text('SELECT 1'))
        return {
            "status": "success",
            "message": "Connexion à la base de données SQLite réussie"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur de connexion à SQLite : {str(e)}"
        }