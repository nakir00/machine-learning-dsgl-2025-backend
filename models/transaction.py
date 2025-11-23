"""
Modèle Transaction - Table des transactions bancaires
"""
from db.connexion.connexion import db
from models.user_auth import UserAuth
from datetime import datetime

from sqlalchemy.orm import Mapped



class Transaction(db.Model):
    """Modèle représentant une transaction bancaire"""
    
    __tablename__ = 'transactions'
    
    # Colonnes
    id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Relation avec l'utilisateur (Foreign Key)
    user_id: int = db.Column(db.Integer, db.ForeignKey('users_auth.id'), nullable=False, index=True)
    
    # Informations client
    gender: int = db.Column(db.Integer, nullable=True)  # 0 = Homme, 1 = Femme
    age: int = db.Column(db.Integer, nullable=True)
    house_type_id: int = db.Column(db.Integer, nullable=True)
    contact_avaliability_id: int = db.Column(db.Integer, nullable=True)
    home_country: int = db.Column(db.Integer, nullable=True)
    
    # Informations compte
    account_no: int = db.Column(db.BigInteger, nullable=True, index=True)
    card_expiry_date: int = db.Column(db.Integer, nullable=True)
    cif: int = db.Column(db.BigInteger, nullable=True)
    
    # Informations transaction
    transaction_amount: float = db.Column(db.Float, nullable=False)
    transaction_country: int = db.Column(db.Integer, nullable=True)
    transaction_currency_code: int = db.Column(db.Integer, nullable=True)
    large_purchase: int = db.Column(db.Integer, default=0)  # 0 = Non, 1 = Oui
    product_id: int = db.Column(db.Integer, nullable=True)
    
    # Fraude
    potential_fraud: int = db.Column(db.Integer, default=0, index=True)  # 0 = Non, 1 = Oui
    prediction: int = db.Column(db.Integer, nullable=True)  # Prédiction du modèle ML
    prediction_proba: float = db.Column(db.Float, nullable=True)  # Probabilité de fraude
    
    # Métadonnées
    created_at: datetime = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at: datetime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relation avec UserAuth (many-to-one)
    user: Mapped['UserAuth'] = db.relationship('UserAuth', back_populates='transactions')
    
    def __repr__(self:'Transaction') -> str:
        """Représentation du modèle"""
        return f'<Transaction {self.id} - Amount: {self.transaction_amount}>'
    
    def to_dict(self:'Transaction', include_all: bool=False, include_user: bool=False) -> dict:
        """
        Convertit l'objet en dictionnaire
        
        Args:
            include_all (bool): Inclure tous les champs ou seulement les essentiels
            include_user (bool): Inclure les infos de l'utilisateur
            
        Returns:
            dict: Représentation de la transaction en JSON
        """
        base_dict = {
            'id': self.id,
            'user_id': self.user_id,
            'account_no': self.account_no,
            'transaction_amount': self.transaction_amount,
            'potential_fraud': self.potential_fraud,
            'prediction': self.prediction,
            'prediction_proba': self.prediction_proba,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        if include_all:
            base_dict.update({
                'gender': self.gender,
                'age': self.age,
                'house_type_id': self.house_type_id,
                'contact_avaliability_id': self.contact_avaliability_id,
                'home_country': self.home_country,
                'card_expiry_date': self.card_expiry_date,
                'cif': self.cif,
                'transaction_country': self.transaction_country,
                'transaction_currency_code': self.transaction_currency_code,
                'large_purchase': self.large_purchase,
                'product_id': self.product_id,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            })
        
        if include_user and self.user:
            base_dict['user'] = {
                'id': self.user.id,
                'username': self.user.username,
                'email': self.user.email
            }
        
        return base_dict
    
    @classmethod
    def from_dict(cls:'Transaction', data: dict, user_id: int=None) -> 'Transaction':
        """
        Crée une instance Transaction depuis un dictionnaire
        
        Args:
            data (dict): Données de la transaction
            user_id (int): ID de l'utilisateur (optionnel si dans data)
            
        Returns:
            Transaction: Instance du modèle Transaction
        """
        return cls(
            user_id=user_id or data.get('user_id'),
            gender=data.get('gender'),
            age=data.get('age'),
            house_type_id=data.get('house_type_id'),
            contact_avaliability_id=data.get('contact_avaliability_id'),
            home_country=data.get('home_country'),
            account_no=data.get('account_no'),
            card_expiry_date=data.get('card_expiry_date'),
            cif=data.get('cif'),
            transaction_amount=data.get('transaction_amount'),
            transaction_country=data.get('transaction_country'),
            transaction_currency_code=data.get('transaction_currency_code'),
            large_purchase=data.get('large_purchase', 0),
            product_id=data.get('product_id'),
            potential_fraud=data.get('potential_fraud', 0),
            prediction=data.get('prediction'),
            prediction_proba=data.get('prediction_proba')
        )