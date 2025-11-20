from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import json
import os
from google.cloud.sql.connector import Connector

# Configuration du port
port = int(os.environ.get("PORT", 8080))

# Initialisation de Flask
app = Flask(__name__)

# ============================================================================
# CONFIGURATION DE LA BASE DE DONNÉES
# ============================================================================

# Variables d'environnement pour Cloud SQL
INSTANCE_CONNECTION_NAME = os.environ.get("INSTANCE_CONNECTION_NAME")  # Format: project:region:instance
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "fraud_detection")

# Variable pour déterminer l'environnement (local ou production)
ENV = os.environ.get("ENV", "production")  # "local" ou "production"

# ============================================================================
# CONFIGURATION SQLALCHEMY
# ============================================================================

if ENV == "production":
    # Configuration pour Cloud Run (avec Cloud SQL Connector)
    print("🚀 Mode PRODUCTION - Connexion à Cloud SQL via Connector")
    
    # Initialiser le connecteur Cloud SQL
    connector = Connector()
    
    def getconn():
        """Crée une connexion à Cloud SQL"""
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pymysql",  # ou "pg8000" pour PostgreSQL
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME
        )
        return conn
    
    # Créer le moteur SQLAlchemy avec le connecteur
    engine = create_engine(
        "mysql+pymysql://",  # ou "postgresql+pg8000://" pour PostgreSQL
        creator=getconn,
        poolclass=NullPool,
    )
    
    app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://"
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'creator': getconn,
        'poolclass': NullPool,
    }

else:
    # Configuration pour développement local
    print("🏠 Mode LOCAL - Connexion à base de données locale")
    
    # Option 1: Base de données locale (MySQL/PostgreSQL)
    LOCAL_DB_HOST = os.environ.get("LOCAL_DB_HOST", "localhost")
    LOCAL_DB_PORT = os.environ.get("LOCAL_DB_PORT", "3306")
    
    DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{LOCAL_DB_HOST}:{LOCAL_DB_PORT}/{DB_NAME}"
    
    # Option 2: SQLite pour développement (plus simple, pas besoin de serveur)
    # DATABASE_URI = "sqlite:///fraud_detection.db"
    
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

# Configuration commune
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True  # Active les logs SQL (désactiver en prod)

# Initialisation de SQLAlchemy
db = SQLAlchemy(app)

# ============================================================================
# DÉFINITION DES MODÈLES (TABLES)
# ============================================================================

class User(db.Model):
    """Modèle d'exemple : Table des utilisateurs"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Transaction(db.Model):
    """Modèle : Table des transactions (pour détection de fraude)"""
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.Integer)
    age = db.Column(db.Integer)
    house_type_id = db.Column(db.Integer)
    contact_avaliability_id = db.Column(db.Integer)
    home_country = db.Column(db.Integer)
    account_no = db.Column(db.Integer)
    card_expiry_date = db.Column(db.Integer)
    transaction_amount = db.Column(db.Float)
    transaction_country = db.Column(db.Integer)
    large_purchase = db.Column(db.Integer)
    product_id = db.Column(db.Integer)
    cif = db.Column(db.Integer)
    transaction_currency_code = db.Column(db.Integer)
    potential_fraud = db.Column(db.Integer)  # 0 = Non, 1 = Oui
    prediction = db.Column(db.Integer, nullable=True)  # Prédiction du modèle
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire"""
        return {
            'id': self.id,
            'gender': self.gender,
            'age': self.age,
            'transaction_amount': self.transaction_amount,
            'potential_fraud': self.potential_fraud,
            'prediction': self.prediction,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ============================================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================================

with app.app_context():
    try:
        # Créer toutes les tables
        db.create_all()
        print("✅ Tables créées avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables : {e}")

# ============================================================================
# ROUTES API
# ============================================================================

@app.route("/")
def hello_world():
    return jsonify({
        "message": "API de détection de fraude bancaire",
        "status": "online",
        "environment": ENV
    })

@app.route("/health")
def health_check():
    """Vérification de santé de l'API et de la connexion DB"""
    try:
        # Tester la connexion à la base de données
        db.session.execute(db.text('SELECT 1'))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "environment": ENV
    })

# ============================================================================
# ROUTES CRUD - USERS (EXEMPLE)
# ============================================================================

@app.route("/users", methods=["GET"])
def get_users():
    """Récupérer tous les utilisateurs"""
    try:
        users = User.query.all()
        return jsonify({
            "success": True,
            "data": [user.to_dict() for user in users]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/users", methods=["POST"])
def create_user():
    """Créer un nouvel utilisateur"""
    try:
        data = request.get_json()
        
        new_user = User(
            username=data['username'],
            email=data['email']
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Utilisateur créé",
            "data": new_user.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Récupérer un utilisateur par ID"""
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({
            "success": True,
            "data": user.to_dict()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 404

# ============================================================================
# ROUTES CRUD - TRANSACTIONS
# ============================================================================

@app.route("/transactions", methods=["GET"])
def get_transactions():
    """Récupérer toutes les transactions"""
    try:
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        transactions = Transaction.query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return jsonify({
            "success": True,
            "data": [t.to_dict() for t in transactions.items],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": transactions.total,
                "pages": transactions.pages
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/transactions", methods=["POST"])
def create_transaction():
    """Créer une nouvelle transaction"""
    try:
        data = request.get_json()
        
        new_transaction = Transaction(
            gender=data.get('gender'),
            age=data.get('age'),
            house_type_id=data.get('house_type_id'),
            contact_avaliability_id=data.get('contact_avaliability_id'),
            home_country=data.get('home_country'),
            account_no=data.get('account_no'),
            card_expiry_date=data.get('card_expiry_date'),
            transaction_amount=data.get('transaction_amount'),
            transaction_country=data.get('transaction_country'),
            large_purchase=data.get('large_purchase'),
            product_id=data.get('product_id'),
            cif=data.get('cif'),
            transaction_currency_code=data.get('transaction_currency_code'),
            potential_fraud=data.get('potential_fraud', 0)
        )
        
        db.session.add(new_transaction)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Transaction créée",
            "data": new_transaction.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/transactions/<int:transaction_id>", methods=["GET"])
def get_transaction(transaction_id):
    """Récupérer une transaction par ID"""
    try:
        transaction = Transaction.query.get_or_404(transaction_id)
        return jsonify({
            "success": True,
            "data": transaction.to_dict()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 404

@app.route("/transactions/fraud", methods=["GET"])
def get_fraud_transactions():
    """Récupérer uniquement les transactions frauduleuses"""
    try:
        frauds = Transaction.query.filter_by(potential_fraud=1).all()
        return jsonify({
            "success": True,
            "count": len(frauds),
            "data": [t.to_dict() for t in frauds]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================================
# GESTION DES ERREURS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Route non trouvée"}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({"success": False, "error": "Erreur serveur interne"}), 500

# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", static_url_path="/static", port=port, debug=(ENV == "local"))