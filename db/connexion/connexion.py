"""
Configuration de la connexion à la base de données SQLite
Support MySQL en option via variable d'environnement
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import QueuePool, StaticPool
import os
from pathlib import Path

# Instance SQLAlchemy (sera initialisée dans app.py)
db = SQLAlchemy()


def init_db(app):
    """
    Initialise la connexion à la base de données
    Support SQLite (par défaut) et MySQL (optionnel)
    
    Args:
        app: Instance Flask
        
    Returns:
        db: Instance SQLAlchemy
    """
    # Déterminer le type de base de données
    db_type = os.environ.get("DB_TYPE", "sqlite").lower()
    
    if db_type == "mysql":
        # Configuration MySQL
        database_url = _configure_mysql()
        pool_config = {
            'pool_size': 10,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'pool_timeout': 30,
            'max_overflow': 5,
            'poolclass': QueuePool,
            'echo': False
        }
    else:
        # Configuration SQLite (par défaut)
        database_url = _configure_sqlite()
        pool_config = {
            'poolclass': StaticPool,  # SQLite utilise StaticPool
            'connect_args': {
                'check_same_thread': False  # Important pour Flask
            },
            'echo': False
        }
    
    # Configuration SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = pool_config
    
    # Initialiser SQLAlchemy avec l'app Flask
    db.init_app(app)
    
    print(f"🔗 Base de données configurée: {db_type.upper()}")
    
    return db


def _configure_sqlite():
    """
    Configure la connexion SQLite
    
    Returns:
        str: URL de connexion SQLite
    """
    # Récupérer le chemin de la base de données
    db_path = os.environ.get("DATABASE_PATH", "fraud_detection.db")
    
    # Si chemin relatif, le placer dans le dossier du projet
    if not os.path.isabs(db_path):
        # Obtenir le répertoire racine du projet (3 niveaux au-dessus)
        base_dir = Path(__file__).resolve().parent.parent.parent
        db_path = base_dir / db_path
    
    # Créer le dossier parent si nécessaire
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construction de l'URL SQLite
    database_url = f"sqlite:///{db_path}"
    
    print(f"   📁 Fichier: {db_path}")
    print(f"   📊 Taille: {db_path.stat().st_size / 1024:.2f} KB" if db_path.exists() else "   📊 Nouvelle base")
    
    return database_url


def _configure_mysql():
    """
    Configure la connexion MySQL
    
    Returns:
        str: URL de connexion MySQL
    """
    # Récupération des variables d'environnement
    mysql_host = os.environ.get("MYSQL_HOST", "localhost")
    mysql_port = os.environ.get("MYSQL_PORT", "3306")
    mysql_user = os.environ.get("MYSQL_USER", "root")
    mysql_password = os.environ.get("MYSQL_PASSWORD", "")
    mysql_database = os.environ.get("MYSQL_DATABASE", "fraud_detection")
    
    # Construction de l'URL de connexion MySQL
    database_url = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}"
    
    print(f"   🌐 Serveur: {mysql_user}@{mysql_host}:{mysql_port}/{mysql_database}")
    
    return database_url


def create_tables(app):
    """
    Crée toutes les tables définies dans les modèles
    
    Args:
        app: Instance Flask
    """
    with app.app_context():
        try:
            db.create_all()
            
            # Compter les tables créées
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            print(f"✅ Tables créées avec succès ({len(tables)} tables)")
            
            if tables:
                print(f"   Tables: {', '.join(tables)}")
                
        except Exception as e:
            print(f"❌ Erreur lors de la création des tables : {e}")
            import traceback
            traceback.print_exc()


def test_connection():
    """
    Teste la connexion à la base de données
    
    Returns:
        dict: Résultat du test avec statut et message
    """
    try:
        # Test simple de connexion
        result = db.session.execute(db.text('SELECT 1'))
        result.close()
        
        # Obtenir des informations sur la base
        db_type = os.environ.get("DB_TYPE", "sqlite").lower()
        
        if db_type == "sqlite":
            db_path = os.environ.get("DATABASE_PATH", "fraud_detection.db")
            if not os.path.isabs(db_path):
                base_dir = Path(__file__).resolve().parent.parent.parent
                db_path = base_dir / db_path
            
            db_exists = Path(db_path).exists()
            db_size = Path(db_path).stat().st_size if db_exists else 0
            
            return {
                "status": "success",
                "message": "Connexion SQLite réussie",
                "details": {
                    "type": "SQLite",
                    "path": str(db_path),
                    "exists": db_exists,
                    "size_bytes": db_size,
                    "size_mb": round(db_size / (1024 * 1024), 2)
                }
            }
        else:
            return {
                "status": "success",
                "message": "Connexion MySQL réussie",
                "details": {
                    "type": "MySQL",
                    "host": os.environ.get("MYSQL_HOST", "localhost"),
                    "database": os.environ.get("MYSQL_DATABASE", "fraud_detection")
                }
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur de connexion : {str(e)}",
            "details": {
                "type": os.environ.get("DB_TYPE", "sqlite").upper(),
                "error": str(e)
            }
        }


def drop_all_tables(app):
    """
    Supprime toutes les tables (ATTENTION: perte de données!)
    Utile pour le développement ou la réinitialisation
    
    Args:
        app: Instance Flask
    """
    with app.app_context():
        try:
            db.drop_all()
            print("⚠️  Toutes les tables ont été supprimées")
        except Exception as e:
            print(f"❌ Erreur lors de la suppression des tables : {e}")


def reset_database(app):
    """
    Réinitialise complètement la base de données
    Supprime et recrée toutes les tables
    
    Args:
        app: Instance Flask
    """
    print("\n⚠️  ATTENTION: Réinitialisation de la base de données!")
    
    with app.app_context():
        try:
            # Supprimer toutes les tables
            db.drop_all()
            print("   ✅ Tables supprimées")
            
            # Recréer les tables
            db.create_all()
            print("   ✅ Tables recréées")
            
            print("✅ Base de données réinitialisée avec succès\n")
            
        except Exception as e:
            print(f"❌ Erreur lors de la réinitialisation : {e}")


def get_database_info():
    """
    Obtient des informations détaillées sur la base de données
    
    Returns:
        dict: Informations sur la base de données
    """
    try:
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        
        info = {
            "type": os.environ.get("DB_TYPE", "sqlite").upper(),
            "tables_count": len(tables),
            "tables": []
        }
        
        # Informations sur chaque table
        for table_name in tables:
            columns = inspector.get_columns(table_name)
            info["tables"].append({
                "name": table_name,
                "columns_count": len(columns),
                "columns": [col["name"] for col in columns]
            })
        
        return info
        
    except Exception as e:
        return {
            "error": str(e)
        }


def backup_database(backup_path=None):
    """
    Crée une sauvegarde de la base de données SQLite
    (Fonctionne uniquement avec SQLite)
    
    Args:
        backup_path (str): Chemin du fichier de backup
        
    Returns:
        dict: Résultat de la sauvegarde
    """
    db_type = os.environ.get("DB_TYPE", "sqlite").lower()
    
    if db_type != "sqlite":
        return {
            "status": "error",
            "message": "La sauvegarde automatique n'est disponible que pour SQLite"
        }
    
    try:
        import shutil
        from datetime import datetime
        
        # Chemin de la base source
        db_path = os.environ.get("DATABASE_PATH", "fraud_detection.db")
        if not os.path.isabs(db_path):
            base_dir = Path(__file__).resolve().parent.parent.parent
            db_path = base_dir / db_path
        
        # Chemin du backup
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_path.stem}_backup_{timestamp}.db"
            backup_path = db_path.parent / backup_path
        
        # Copier le fichier
        shutil.copy2(db_path, backup_path)
        
        return {
            "status": "success",
            "message": "Sauvegarde créée avec succès",
            "backup_path": str(backup_path),
            "size_mb": round(Path(backup_path).stat().st_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors de la sauvegarde : {str(e)}"
        }