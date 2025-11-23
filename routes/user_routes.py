"""
Routes pour les utilisateurs
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.user import User
from services.user_service import UserService
from validators.user_validator import validate_user_data

# Créer un Blueprint pour les routes user
user_bp = Blueprint('users', __name__, url_prefix='/users')


@user_bp.route('', methods=['GET'])
@jwt_required()
def get_users():
    """Récupérer tous les utilisateurs avec pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)  # Limiter à 100 max
        
        result = UserService.get_all_users(page, per_page)
        
        return jsonify({
            'success': True,
            'count': len(result['users']),
            'data': result['users'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@user_bp.route('/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    """Récupérer un utilisateur par ID"""
    try:
        user: User = UserService.get_user_by_id(user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'Utilisateur non trouvé'
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


@user_bp.route('', methods=['POST'])
@jwt_required()
def create_user():
    """Créer un nouvel utilisateur"""
    try:
        data: dict = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données
        is_valid, errors = validate_user_data(data, is_update=False)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Créer l'utilisateur
        user, error = UserService.create_user(data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Utilisateur créé avec succès',
            'data': user.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@user_bp.route('/<int:user_id>', methods=['PUT', 'PATCH'])
@jwt_required()
def update_user(user_id):
    """Mettre à jour un utilisateur"""
    try:
        data: dict = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données
        is_valid, errors = validate_user_data(data, is_update=True)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Mettre à jour l'utilisateur
        user, error = UserService.update_user(user_id, data)
        
        if error:
            status_code = 404 if "non trouvé" in error else 400
            return jsonify({
                'success': False,
                'error': error
            }), status_code
        
        return jsonify({
            'success': True,
            'message': 'Utilisateur mis à jour avec succès',
            'data': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@user_bp.route('/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """Supprimer un utilisateur"""
    try:
        success, error = UserService.delete_user(user_id)
        
        if not success:
            status_code = 404 if "non trouvé" in error else 400
            return jsonify({
                'success': False,
                'error': error
            }), status_code
        
        return jsonify({
            'success': True,
            'message': 'Utilisateur supprimé avec succès'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@user_bp.route('/search', methods=['GET'])
@jwt_required()
def search_users():
    """Rechercher des utilisateurs"""
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Paramètre de recherche "q" requis'
            }), 400
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)
        
        result = UserService.search_users(query, page, per_page)
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(result['users']),
            'data': result['users'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@user_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_user_stats():
    """Obtenir des statistiques sur les utilisateurs"""
    try:
        total = UserService.get_total_users_count()
        active = UserService.get_active_users_count()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': total,
                'active_users': active,
                'inactive_users': total - active,
                'active_rate': round((active / total * 100), 2) if total > 0 else 0
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500