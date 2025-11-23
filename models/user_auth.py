"""
Modèle UserAuth - Utilisateur avec authentification
"""
from db.connexion.connexion import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


class UserAuth(db.Model):
    """Modèle représentant un utilisateur authentifié"""
    
    __tablename__ = 'users_auth'
    
    # Colonnes
    id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email: str = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username: str = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash: str = db.Column(db.String(256), nullable=False)
    
    # Informations supplémentaires
    first_name: str = db.Column(db.String(100), nullable=True)
    last_name: str = db.Column(db.String(100), nullable=True)
    
    # Statut
    is_active: bool = db.Column(db.Boolean, default=True)
    
    # Métadonnées
    created_at: datetime = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at: datetime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    last_login: datetime = db.Column(db.DateTime, nullable=True)
    
    # Relation One-to-Many avec Transaction
    transactions= db.relationship('Transaction', back_populates='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self:'UserAuth') -> str:
        return f'<UserAuth {self.username}>'
    
    def set_password(self:'UserAuth', password: str) -> None:
        """Hash et stocke le mot de passe"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
    
    def check_password(self:'UserAuth', password: str) -> bool:
        """Vérifie si le mot de passe est correct"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self:'UserAuth') -> None:
        """Met à jour la date de dernière connexion"""
        self.last_login = datetime.now()
    
    def to_dict(self:'UserAuth', include_transactions: bool=False) -> dict:
        """
        Convertit l'objet en dictionnaire (sans le password)
        
        Args:
            include_transactions (bool): Inclure les transactions de l'utilisateur
        """
        data = {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'transactions_count': self.transactions.count()
        }
        
        if include_transactions:
            data['transactions'] = [t.to_dict() for t in self.transactions.limit(100).all()]
        
        return data
    
    @classmethod
    def find_by_email(cls:'UserAuth', email: str) -> 'UserAuth':
        """Trouve un utilisateur par email"""
        return cls.query.filter_by(email=email).first()
    
    @classmethod
    def find_by_username(cls:'UserAuth', username: str) -> 'UserAuth':
        """Trouve un utilisateur par username"""
        return cls.query.filter_by(username=username).first()
    
    @classmethod
    def find_by_id(cls:'UserAuth', user_id: int) -> 'UserAuth':
        """Trouve un utilisateur par ID"""
        return cls.query.get(user_id)