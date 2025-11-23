"""
Configuration JWT pour Flask-JWT-Extended
"""
import os
from datetime import timedelta
from flask_jwt_extended import JWTManager
from models.user_auth import UserAuth
from services.auth_service import AuthService


def init_jwt(app):
    """
    Initialise et configure Flask-JWT-Extended
    
    Args:
        app: Instance Flask
    
    Returns:
        JWTManager: Instance du gestionnaire JWT
    """
    
    # ============================================================================
    # CONFIGURATION JWT (Variables d'environnement)
    # ============================================================================
    
    # Clé secrète pour signer les tokens (OBLIGATOIRE en production)
    app.config['JWT_SECRET_KEY'] = os.environ.get(
        'JWT_SECRET_KEY', 
        'super-secret-key-change-in-production'
    )
    
    # Durée de validité de l'access token (défaut: 15 minutes)
    access_token_minutes = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES_MINUTES', 15))
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=access_token_minutes)
    
    # Durée de validité du refresh token (défaut: 30 jours)
    refresh_token_days = int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES_DAYS', 30))
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=refresh_token_days)
    
    # Localisation du token (header par défaut)
    app.config['JWT_TOKEN_LOCATION'] = ['headers']
    
    # Header attendu: Authorization: Bearer <token>
    app.config['JWT_HEADER_NAME'] = 'Authorization'
    app.config['JWT_HEADER_TYPE'] = 'Bearer'
    
    # Activer la blacklist
    app.config['JWT_BLACKLIST_ENABLED'] = True
    app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
    
    # Initialiser JWTManager
    jwt = JWTManager(app)
    
    # ============================================================================
    # CALLBACKS JWT
    # ============================================================================
    
    @jwt.user_identity_loader
    def user_identity_lookup(user):
        """
        Définit ce qui est stocké dans le token comme identité
        Appelé lors de la création du token
        """
        if isinstance(user, UserAuth):
            return user.id
        return user
    
    @jwt.user_lookup_loader
    def user_lookup_callback(_jwt_header, jwt_data):
        """
        Charge l'utilisateur depuis l'identité du token
        Permet d'utiliser current_user dans les routes
        """
        identity = jwt_data['sub']
        # Convertir en int car l'identity est stockée en string
        return UserAuth.find_by_id(int(identity))
    
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        """
        Vérifie si le token est dans la blacklist (révoqué)
        Appelé à chaque requête avec un token
        """
        jti = jwt_payload['jti']
        return AuthService.is_token_revoked(jti)
    
    # ============================================================================
    # GESTIONNAIRES D'ERREURS JWT
    # ============================================================================
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        """Token expiré"""
        return {
            'success': False,
            'error': 'Token expiré',
            'code': 'token_expired'
        }, 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        """Token invalide"""
        return {
            'success': False,
            'error': 'Token invalide',
            'code': 'invalid_token'
        }, 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        """Token manquant"""
        return {
            'success': False,
            'error': 'Token d\'authentification requis',
            'code': 'authorization_required'
        }, 401
    
    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        """Token révoqué (dans la blacklist)"""
        return {
            'success': False,
            'error': 'Token révoqué. Veuillez vous reconnecter.',
            'code': 'token_revoked'
        }, 401
    
    @jwt.needs_fresh_token_loader
    def token_not_fresh_callback(jwt_header, jwt_payload):
        """Token pas assez récent (pour les opérations sensibles)"""
        return {
            'success': False,
            'error': 'Token frais requis. Veuillez vous reconnecter.',
            'code': 'fresh_token_required'
        }, 401
    
    # Afficher la configuration
    print(f"🔐 JWT configuré :")
    print(f"   Access Token  : {access_token_minutes} minutes")
    print(f"   Refresh Token : {refresh_token_days} jours")
    
    return jwt