"""
Validateurs pour les requêtes Transaction avec Flask-WTF
"""
from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField
from wtforms.validators import DataRequired, Optional, NumberRange, ValidationError

import logging
logging.basicConfig(level=logging.INFO)  # ou DEBUG
logger = logging.getLogger(__name__)

class CreateTransactionForm(FlaskForm):
    """Formulaire de création de transaction"""
    
    # Informations client
    gender = IntegerField('Gender', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Gender doit être 0 ou 1")
    ])
    
    age = IntegerField('Age', validators=[
        Optional(),
        NumberRange(min=0, max=150, message="Age doit être entre 0 et 150")
    ])
    
    house_type_id = IntegerField('House Type ID', validators=[Optional()])
    contact_avaliability_id = IntegerField('Contact Availability ID', validators=[Optional()])
    home_country = IntegerField('Home Country', validators=[Optional()])
    
    # Informations compte
    account_no = IntegerField('Account Number', validators=[Optional()])
    card_expiry_date = IntegerField('Card Expiry Date', validators=[Optional()])
    cif = IntegerField('CIF', validators=[Optional()])
    
    # Informations transaction (OBLIGATOIRE)
    transaction_amount = FloatField('Transaction Amount', validators=[
        DataRequired(message="Le montant de la transaction est requis"),
        NumberRange(min=0, message="Le montant doit être positif")
    ])
    
    transaction_country = IntegerField('Transaction Country', validators=[Optional()])
    transaction_currency_code = IntegerField('Transaction Currency Code', validators=[Optional()])
    
    large_purchase = IntegerField('Large Purchase', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Large Purchase doit être 0 ou 1")
    ])
    
    product_id = IntegerField('Product ID', validators=[Optional()])
    
    # Fraude
    potential_fraud = IntegerField('Potential Fraud', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Potential Fraud doit être 0 ou 1")
    ])
    
    prediction = IntegerField('Prediction', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Prediction doit être 0 ou 1")
    ])
    
    prediction_proba = FloatField('Prediction Probability', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="La probabilité doit être entre 0 et 1")
    ])
    
    def validate_transaction_amount(self, field):
        """Validation personnalisée pour le montant"""
        if field.data is not None and field.data > 1000000:
            raise ValidationError("Le montant ne peut pas dépasser 1,000,000")


class UpdateTransactionForm(FlaskForm):
    """Formulaire de mise à jour de transaction"""
    
    # Tous les champs sont optionnels pour une mise à jour
    transaction_amount = FloatField('Transaction Amount', validators=[
        Optional(),
        NumberRange(min=0, message="Le montant doit être positif")
    ])
    
    potential_fraud = IntegerField('Potential Fraud', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Potential Fraud doit être 0 ou 1")
    ])
    
    prediction = IntegerField('Prediction', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="Prediction doit être 0 ou 1")
    ])
    
    prediction_proba = FloatField('Prediction Probability', validators=[
        Optional(),
        NumberRange(min=0, max=1, message="La probabilité doit être entre 0 et 1")
    ])


def validate_transaction_data(data, is_update=False, for_prediction=False ):
    """
    Fonction helper pour valider les données transaction (alternative sans WTForms)
    
    Args:
        data (dict): Données à valider
        is_update (bool): True si c'est une mise à jour
        
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    # Validation pour création
    if not is_update:
        if 'transaction_amount' not in data or data['transaction_amount'] is None:
            errors.append("Le montant de la transaction est requis")
        elif data['transaction_amount'] < 0:
            errors.append("Le montant doit être positif")
        elif data['transaction_amount'] > 1000000:
            errors.append("Le montant ne peut pas dépasser 1,000,000")
    
    # Validation des champs numériques
    if 'gender' in data and data['gender'] is not None:
        if data['gender'] not in [0, 1]:
            errors.append("Gender doit être 0 ou 1")
    
    if 'age' in data and data['age'] is not None:
        if not 0 <= data['age'] <= 150:
            errors.append("Age doit être entre 0 et 150")
    
    if 'large_purchase' in data and data['large_purchase'] is not None:
        if data['large_purchase'] not in [0, 1]:
            errors.append("Large Purchase doit être 0 ou 1")
    
    if 'potential_fraud' in data and data['potential_fraud'] is not None:
        if data['potential_fraud'] not in [0, 1]:
            errors.append("Potential Fraud doit être 0 ou 1")
    
    if 'prediction' in data and data['prediction'] is not None:
        if data['prediction'] not in [0, 1]:
            errors.append("Prediction doit être 0 ou 1")
    
    if 'prediction_proba' in data and data['prediction_proba'] is not None:
        if not 0 <= data['prediction_proba'] <= 1:
            errors.append("La probabilité doit être entre 0 et 1")
    
    return len(errors) == 0, errors


def validate_fraud_prediction(transaction_data):
    """
    Validation spécifique pour les prédictions de fraude
    
    Args:
        transaction_data (dict): Données de la transaction
        
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Vérifier que les champs nécessaires pour la prédiction sont présents
    required_fields = [
        'gender', 'age', 'house_type_id', 'contact_avaliability_id',
        'home_country', 'account_no', 'card_expiry_date', 'transaction_amount',
        'transaction_country', 'large_purchase', 'product_id', 'cif',
        'transaction_currency_code'
    ]
    
    missing_fields = [field for field in required_fields if field not in transaction_data]
    
    if missing_fields:
        errors.append(f"Champs manquants pour la prédiction : {', '.join(missing_fields)}")
    
    return len(errors) == 0, errors