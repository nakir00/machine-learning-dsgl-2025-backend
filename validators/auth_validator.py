"""
Validateurs pour l'authentification
"""
import re


def validate_register_data(data: dict):
    """
    Valide les données d'inscription
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Email requis et valide
    if not data.get('email'):
        errors.append("L'email est requis")
    elif not is_valid_email(data['email']):
        errors.append("Format d'email invalide")
    elif len(data['email']) > 120:
        errors.append("L'email ne peut pas dépasser 120 caractères")
    
    # Username requis
    if not data.get('username'):
        errors.append("Le nom d'utilisateur est requis")
    elif len(data['username']) < 3:
        errors.append("Le nom d'utilisateur doit contenir au moins 3 caractères")
    elif len(data['username']) > 80:
        errors.append("Le nom d'utilisateur ne peut pas dépasser 80 caractères")
    elif not is_valid_username(data['username']):
        errors.append("Le nom d'utilisateur ne peut contenir que des lettres, chiffres et underscores")
    
    # Password requis et fort
    if not data.get('password'):
        errors.append("Le mot de passe est requis")
    else:
        password_errors = validate_password_strength(data['password'])
        errors.extend(password_errors)
    
    # Confirmation du mot de passe (optionnel)
    if data.get('password_confirm') and data.get('password'):
        if data['password'] != data['password_confirm']:
            errors.append("Les mots de passe ne correspondent pas")
    
    return len(errors) == 0, errors


def validate_login_data(data: dict):
    """
    Valide les données de connexion
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    if not data.get('email'):
        errors.append("L'email est requis")
    
    if not data.get('password'):
        errors.append("Le mot de passe est requis")
    
    return len(errors) == 0, errors


def validate_change_password_data(data: dict):
    """
    Valide les données de changement de mot de passe
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    if not data.get('current_password'):
        errors.append("Le mot de passe actuel est requis")
    
    if not data.get('new_password'):
        errors.append("Le nouveau mot de passe est requis")
    else:
        password_errors = validate_password_strength(data['new_password'])
        errors.extend(password_errors)
    
    if data.get('new_password') and data.get('current_password'):
        if data['new_password'] == data['current_password']:
            errors.append("Le nouveau mot de passe doit être différent de l'ancien")
    
    return len(errors) == 0, errors


def validate_password_strength(password: str):
    """
    Vérifie la force du mot de passe
    
    Returns:
        list: Liste des erreurs (vide si valide)
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Le mot de passe doit contenir au moins 8 caractères")
    
    if len(password) > 128:
        errors.append("Le mot de passe ne peut pas dépasser 128 caractères")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Le mot de passe doit contenir au moins une majuscule")
    
    if not re.search(r'[a-z]', password):
        errors.append("Le mot de passe doit contenir au moins une minuscule")
    
    if not re.search(r'\d', password):
        errors.append("Le mot de passe doit contenir au moins un chiffre")
    
    return errors


def is_valid_email(email: str):
    """Vérifie si l'email a un format valide"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_username(username: str):
    """Vérifie si le username est valide (lettres, chiffres, underscores)"""
    pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(pattern, username) is not None