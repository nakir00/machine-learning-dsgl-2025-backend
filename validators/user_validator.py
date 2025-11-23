"""
Validateurs pour les requêtes User avec Flask-WTF
"""
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired, Email, Length, Optional, ValidationError
from models.user import User


class CreateUserForm(FlaskForm):
    """Formulaire de création d'utilisateur"""
    
    username = StringField('Username', validators=[
        DataRequired(message="Le nom d'utilisateur est requis"),
        Length(min=3, max=80, message="Le nom d'utilisateur doit contenir entre 3 et 80 caractères")
    ])
    
    email = StringField('Email', validators=[
        DataRequired(message="L'email est requis"),
        Email(message="Format d'email invalide"),
        Length(max=120, message="L'email ne peut pas dépasser 120 caractères")
    ])
    
    first_name = StringField('First Name', validators=[
        Optional(),
        Length(max=100, message="Le prénom ne peut pas dépasser 100 caractères")
    ])
    
    last_name = StringField('Last Name', validators=[
        Optional(),
        Length(max=100, message="Le nom ne peut pas dépasser 100 caractères")
    ])
    
    is_active = BooleanField('Is Active', validators=[Optional()], default=True)
    
    def validate_username(self, field):
        """Vérifie que le username n'existe pas déjà"""
        if User.query.filter_by(username=field.data).first():
            raise ValidationError("Ce nom d'utilisateur existe déjà")
    
    def validate_email(self, field):
        """Vérifie que l'email n'existe pas déjà"""
        if User.query.filter_by(email=field.data).first():
            raise ValidationError("Cet email est déjà utilisé")


class UpdateUserForm(FlaskForm):
    """Formulaire de mise à jour d'utilisateur"""
    
    username = StringField('Username', validators=[
        Optional(),
        Length(min=3, max=80, message="Le nom d'utilisateur doit contenir entre 3 et 80 caractères")
    ])
    
    email = StringField('Email', validators=[
        Optional(),
        Email(message="Format d'email invalide"),
        Length(max=120, message="L'email ne peut pas dépasser 120 caractères")
    ])
    
    first_name = StringField('First Name', validators=[
        Optional(),
        Length(max=100, message="Le prénom ne peut pas dépasser 100 caractères")
    ])
    
    last_name = StringField('Last Name', validators=[
        Optional(),
        Length(max=100, message="Le nom ne peut pas dépasser 100 caractères")
    ])
    
    is_active = BooleanField('Is Active', validators=[Optional()])


def validate_user_data(data, is_update=False):
    """
    Fonction helper pour valider les données user (alternative sans WTForms)
    
    Args:
        data (dict): Données à valider
        is_update (bool): True si c'est une mise à jour
        
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Validation pour création
    if not is_update:
        if not data.get('username'):
            errors.append("Le nom d'utilisateur est requis")
        elif len(data.get('username', '')) < 3:
            errors.append("Le nom d'utilisateur doit contenir au moins 3 caractères")
        
        if not data.get('email'):
            errors.append("L'email est requis")
        elif '@' not in data.get('email', ''):
            errors.append("Format d'email invalide")
        
        # Vérifier l'unicité
        if data.get('username') and User.query.filter_by(username=data['username']).first():
            errors.append("Ce nom d'utilisateur existe déjà")
        
        if data.get('email') and User.query.filter_by(email=data['email']).first():
            errors.append("Cet email est déjà utilisé")
    
    # Validation longueur des champs
    if data.get('username') and len(data['username']) > 80:
        errors.append("Le nom d'utilisateur ne peut pas dépasser 80 caractères")
    
    if data.get('email') and len(data['email']) > 120:
        errors.append("L'email ne peut pas dépasser 120 caractères")
    
    if data.get('first_name') and len(data['first_name']) > 100:
        errors.append("Le prénom ne peut pas dépasser 100 caractères")
    
    if data.get('last_name') and len(data['last_name']) > 100:
        errors.append("Le nom ne peut pas dépasser 100 caractères")
    
    return len(errors) == 0, errors