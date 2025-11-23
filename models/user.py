"""
Modèle User - Table des utilisateurs
"""
from db.connexion.connexion import db
from datetime import datetime


class User(db.Model):
    """Modèle représentant un utilisateur"""
    
    __tablename__ = 'users'
    
    # Colonnes
    id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username: str = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email: str = db.Column(db.String(120), unique=True, nullable=False, index=True)
    first_name: str = db.Column(db.String(100), nullable=True)
    last_name: str = db.Column(db.String(100), nullable=True)
    is_active: bool = db.Column(db.Boolean, default=True)
    created_at: datetime = db.Column(db.DateTime, default=datetime.now(), nullable=False)
    updated_at: datetime = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())
    
    def __repr__(self):
        """Représentation du modèle"""
        return f'<User {self.username}>'
    
    def to_dict(self):
        """
        Convertit l'objet en dictionnaire
        
        Returns:
            dict: Représentation du user en JSON
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data)->'User':
        """
        Crée une instance User depuis un dictionnaire
        
        Args:
            data (dict): Données de l'utilisateur
            
        Returns:
            User: Instance du modèle User
        """
        return cls(
            username=data.get('username'),
            email=data.get('email'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name'),
            is_active=data.get('is_active', True)
        )