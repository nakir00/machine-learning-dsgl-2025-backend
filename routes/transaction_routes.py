from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.transaction_service import TransactionService
from validators.transaction_validator import validate_transaction_data

# Créer un Blueprint pour les routes transaction
transaction_bp = Blueprint('transactions', __name__, url_prefix='/transactions')


@transaction_bp.route('', methods=['GET'])
@jwt_required()
def get_transactions():
    """Récupérer toutes les transactions avec pagination"""
    try:
        page: int = request.args.get('page', 1, type=int)
        per_page: int = request.args.get('per_page', 20, type=int)
        per_page: int = min(per_page, 100)  # Limiter à 100 max
        
        result = TransactionService.get_all_transactions(page, per_page)
        
        return jsonify({
            'success': True,
            'count': len(result['transactions']),
            'data': result['transactions'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/my', methods=['GET'])
@jwt_required()
def get_my_transactions():
    """Récupérer les transactions de l'utilisateur connecté uniquement"""
    try:
        user_id: int = int(get_jwt_identity())
        page: int = request.args.get('page', 1, type=int)
        per_page: int = request.args.get('per_page', 20, type=int)
        per_page: int = min(per_page, 100)
        
        result = TransactionService.get_transactions_by_user(user_id, page, per_page)
        
        return jsonify({
            'success': True,
            'count': len(result['transactions']),
            'data': result['transactions'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/my/stats', methods=['GET'])
@jwt_required()
def get_my_transaction_stats():
    """Statistiques des transactions de l'utilisateur connecté"""
    try:
        user_id = int(get_jwt_identity())
        
        stats = TransactionService.get_user_transaction_statistics(user_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/<int:transaction_id>', methods=['GET'])
@jwt_required()
def get_transaction(transaction_id: int):
    """Récupérer une transaction par ID"""
    try:
        transaction = TransactionService.get_transaction_by_id(transaction_id)
        
        if not transaction:
            return jsonify({
                'success': False,
                'error': 'Transaction non trouvée'
            }), 404
        
        return jsonify({
            'success': True,
            'data': transaction.to_dict(include_all=True)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('', methods=['POST'])
@jwt_required()
def create_transaction():
    """Créer une nouvelle transaction pour l'utilisateur connecté"""
    try:
        user_id: int = int(get_jwt_identity())
        data: dict = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données
        is_valid, errors = validate_transaction_data(data, is_update=False)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Ajouter l'user_id automatiquement
        data['user_id'] = user_id
        
        # Créer la transaction
        transaction, error = TransactionService.create_transaction(data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Transaction créée avec succès',
            'data': transaction.to_dict(include_all=True)
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/<int:transaction_id>', methods=['PUT', 'PATCH'])
@jwt_required()
def update_transaction(transaction_id: int):
    """Mettre à jour une transaction"""
    try:
        data: dict = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données
        is_valid, errors = validate_transaction_data(data, is_update=True)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Mettre à jour la transaction
        transaction, error = TransactionService.update_transaction(transaction_id, data)
        
        if error:
            status_code = 404 if "non trouvée" in error else 400
            return jsonify({
                'success': False,
                'error': error
            }), status_code
        
        return jsonify({
            'success': True,
            'message': 'Transaction mise à jour avec succès',
            'data': transaction.to_dict(include_all=True)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/<int:transaction_id>', methods=['DELETE'])
@jwt_required()
def delete_transaction(transaction_id):
    """Supprimer une transaction"""
    try:
        success, error = TransactionService.delete_transaction(transaction_id)
        
        if not success:
            status_code = 404 if "non trouvée" in error else 400
            return jsonify({
                'success': False,
                'error': error
            }), status_code
        
        return jsonify({
            'success': True,
            'message': 'Transaction supprimée avec succès'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/fraud', methods=['GET'])
@jwt_required()
def get_fraud_transactions():
    """Récupérer uniquement les transactions frauduleuses"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)
        
        result = TransactionService.get_fraud_transactions(page, per_page)
        
        return jsonify({
            'success': True,
            'count': len(result['transactions']),
            'data': result['transactions'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/account/<int:account_no>', methods=['GET'])
@jwt_required()
def get_transactions_by_account(account_no: int):
    """Récupérer les transactions d'un compte"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)
        
        result = TransactionService.get_transactions_by_account(account_no, page, per_page)
        
        return jsonify({
            'success': True,
            'account_no': account_no,
            'count': len(result['transactions']),
            'data': result['transactions'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_transaction_stats():
    """Obtenir des statistiques sur les transactions"""
    try:
        stats = TransactionService.get_transaction_statistics()
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/search', methods=['GET'])
@jwt_required()
def search_transactions():
    """Rechercher des transactions par montant"""
    try:
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)
        
        result = TransactionService.search_transactions_by_amount(
            min_amount, max_amount, page, per_page
        )
        
        return jsonify({
            'success': True,
            'filters': {
                'min_amount': min_amount,
                'max_amount': max_amount
            },
            'count': len(result['transactions']),
            'data': result['transactions'],
            'pagination': result['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@transaction_bp.route('/<int:transaction_id>/mark-fraud', methods=['POST'])
@jwt_required()
def mark_transaction_as_fraud(transaction_id:int):
    """Marquer une transaction comme frauduleuse"""
    try:
        data = request.get_json() or {}
        is_fraud = data.get('is_fraud', True)
        
        transaction, error = TransactionService.mark_as_fraud(transaction_id, is_fraud)
        
        if error:
            status_code = 404 if "non trouvée" in error else 400
            return jsonify({
                'success': False,
                'error': error
            }), status_code
        
        return jsonify({
            'success': True,
            'message': f"Transaction marquée comme {'frauduleuse' if is_fraud else 'légitime'}",
            'data': transaction.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500