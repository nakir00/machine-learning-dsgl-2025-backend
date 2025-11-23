"""
Modèle TokenBlacklist - Stockage des tokens révoqués
"""
from db.connexion.connexion import db
from datetime import datetime

from models.user_auth import UserAuth
from sqlalchemy.orm import Mapped


class TokenBlacklist(db.Model):
    """Modèle pour stocker les tokens révoqués (logout)"""
    
    __tablename__ = 'token_blacklist'
    
    id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    jti: str = db.Column(db.String(36), nullable=False, unique=True, index=True)  # JWT ID unique
    token_type: str = db.Column(db.String(10), nullable=False)  # 'access' ou 'refresh'
    user_id: int = db.Column(db.Integer, db.ForeignKey('users_auth.id'), nullable=False)
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at: datetime = db.Column(db.DateTime, nullable=False)
    
    # Relation avec l'utilisateur
    user: Mapped["UserAuth"] = db.relationship('UserAuth', backref=db.backref('blacklisted_tokens', lazy='dynamic'))
    
    def __repr__(self: 'TokenBlacklist') -> str:
        return f'<TokenBlacklist {self.jti}>'
    
    @classmethod
    def is_token_blacklisted(cls: 'TokenBlacklist', jti: str) -> bool:
        """Vérifie si un token est dans la blacklist"""
        return cls.query.filter_by(jti=jti).first() is not None
    
    @classmethod
    def add_token(cls: 'TokenBlacklist', jti: str, token_type: str, user_id: int, expires_at: datetime) -> 'TokenBlacklist':
        """Ajoute un token à la blacklist"""
        token = cls(
            jti=jti,
            token_type=token_type,
            user_id=user_id,
            expires_at=expires_at
        )
        db.session.add(token)
        db.session.commit()
        return token
    
    @classmethod
    def cleanup_expired(cls: 'TokenBlacklist') -> int:
        """Supprime les tokens expirés de la blacklist (nettoyage)"""
        now = datetime.utcnow()
        expired = cls.query.filter(cls.expires_at < now).all()
        count = len(expired)
        for token in expired:
            db.session.delete(token)
        db.session.commit()
        return count
    
    @classmethod
    def revoke_all_user_tokens(cls: 'TokenBlacklist', user_id: int) -> None:
        """Révoque tous les tokens d'un utilisateur (déconnexion de tous les appareils)"""
        # Note: Cette méthode marque tous les tokens existants comme révoqués
        # En pratique, on pourrait stocker les tokens actifs plutôt que blacklistés
        pass  # Implémentation future si nécessaire