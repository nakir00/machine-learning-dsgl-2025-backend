"""
Routes d'authentification JWT
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    jwt_required, 
    get_jwt_identity, 
    get_jwt,
    current_user
)
from services.auth_service import AuthService
from validators.auth_validator import (
    validate_register_data,
    validate_login_data,
    validate_change_password_data
)
from datetime import datetime, timezone

# Créer le Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Inscription d'un nouvel utilisateur
    
    Body JSON:
    {
        "email": "user@example.com",
        "username": "johndoe",
        "password": "SecurePass123",
        "first_name": "John",      // optionnel
        "last_name": "Doe"         // optionnel
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation
        is_valid, errors = validate_register_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Créer l'utilisateur
        user, error = AuthService.register(
            email=data['email'],
            username=data['username'],
            password=data['password'],
            first_name=data.get('first_name'),
            last_name=data.get('last_name')
        )
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Compte créé avec succès',
            'data': user.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Connexion d'un utilisateur
    
    Body JSON:
    {
        "email": "user@example.com",
        "password": "SecurePass123"
    }
    
    Response:
    {
        "success": true,
        "data": {
            "access_token": "...",
            "refresh_token": "...",
            "user": {...}
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation
        is_valid, errors = validate_login_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Authentification
        tokens, error = AuthService.login(
            email=data['email'],
            password=data['password']
        )
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 401
        
        return jsonify({
            'success': True,
            'message': 'Connexion réussie',
            'data': tokens
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """
    Rafraîchir le access token avec le refresh token
    
    Header: Authorization: Bearer <refresh_token>
    
    Response:
    {
        "success": true,
        "data": {
            "access_token": "..."
        }
    }
    """
    try:
        user_id = int(get_jwt_identity())  # Convertir en int
        
        tokens, error = AuthService.refresh_token(user_id)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 401
        
        return jsonify({
            'success': True,
            'data': tokens
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    Déconnexion (révoque le token actuel)
    
    Header: Authorization: Bearer <access_token>
    """
    try:
        jwt = get_jwt()
        jti = jwt['jti']
        token_type = jwt['type']
        user_id = get_jwt_identity()
        expires_at = datetime.fromtimestamp(jwt['exp'], tz=timezone.utc)
        
        success, error = AuthService.logout(
            jti=jti,
            token_type=token_type,
            user_id=user_id,
            expires_at=expires_at
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Déconnexion réussie'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Récupérer le profil de l'utilisateur connecté
    
    Header: Authorization: Bearer <access_token>
    """
    try:
        user_id = get_jwt_identity()
        
        user, error = AuthService.get_user_profile(user_id)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 404
        
        return jsonify({
            'success': True,
            'data': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/me', methods=['PUT', 'PATCH'])
@jwt_required()
def update_profile():
    """
    Mettre à jour le profil de l'utilisateur connecté
    
    Header: Authorization: Bearer <access_token>
    
    Body JSON:
    {
        "first_name": "John",
        "last_name": "Doe",
        "username": "johndoe"
    }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        user, error = AuthService.update_profile(user_id, data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Profil mis à jour',
            'data': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """
    Changer le mot de passe
    
    Header: Authorization: Bearer <access_token>
    
    Body JSON:
    {
        "current_password": "OldPass123",
        "new_password": "NewSecurePass456"
    }
    """
    try:
        user_id: int = get_jwt_identity()
        data: dict = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation
        is_valid, errors = validate_change_password_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        success, error = AuthService.change_password(
            user_id=user_id,
            current_password=data['current_password'],
            new_password=data['new_password']
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Mot de passe modifié avec succès'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500