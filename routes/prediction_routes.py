"""
Routes pour les prédictions de fraude
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.prediction_service import PredictionService
from services.transaction_service import TransactionService
from validators.transaction_validator import validate_fraud_prediction

# Créer un Blueprint pour les routes de prédiction
prediction_bp = Blueprint('predictions', __name__, url_prefix='/predict')

# Initialiser le service de prédiction
prediction_service = PredictionService()


@prediction_bp.route('/status', methods=['GET'])
@jwt_required()
def model_status():
    """Vérifier le statut du modèle ML"""
    try:
        info = PredictionService.get_model_info()
        
        return jsonify({
            'success': True,
            'model': info
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/reload', methods=['POST'])
@jwt_required()
def reload_model():
    """Recharger le modèle ML depuis les fichiers"""
    try:
        data = request.get_json() or {}
        
        model_path = data.get('model_path')
        scaler_path = data.get('scaler_path')
        stats_path = data.get('stats_path')
        
        success = PredictionService.load_model(model_path, scaler_path, stats_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Modèle rechargé avec succès',
                'model': PredictionService.get_model_info()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Échec du rechargement du modèle'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/transaction', methods=['POST'])
@jwt_required()
def predict_transaction():
    """
    Prédire si une transaction est frauduleuse
    
    Body JSON:
    {
        "gender": 0,
        "age": 35,
        ...
        "use_rules": true,      // Optionnel: utiliser les règles métier (défaut: true)
        "rules_weight": 0.3     // Optionnel: poids des règles 0-1 (défaut: 0.3)
    }
    """
    try:
        data = request.get_json()
        print(data)
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données
        is_valid, errors = validate_fraud_prediction(data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Faire la prédiction
        result = PredictionService.predict(data)
        
        if not result['success']:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/transaction/explain', methods=['POST'])
@jwt_required()
def explain_transaction():
    """
    Analyser les facteurs de risque d'une transaction
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée fournie'
            }), 400
        
        # Faire la prédiction
        prediction_result = PredictionService.predict(data)
        
        # Expliquer les facteurs de risque
        explanation = PredictionService.explain_prediction(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction_result if prediction_result['success'] else None,
            'analysis': explanation
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/batch', methods=['POST'])
@jwt_required()
def predict_batch():
    """
    Prédire pour un batch de transactions
    
    Body JSON:
    {
        "transactions": [
            {"gender": 0, "age": 35, "transaction_amount": 150.50, ...},
            {"gender": 1, "age": 42, "transaction_amount": 2500.00, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({
                'success': False,
                'error': 'Liste de transactions requise (clé: "transactions")'
            }), 400
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({
                'success': False,
                'error': '"transactions" doit être une liste'
            }), 400
        
        if len(transactions) == 0:
            return jsonify({
                'success': False,
                'error': 'La liste de transactions est vide'
            }), 400
        
        if len(transactions) > 1000:
            return jsonify({
                'success': False,
                'error': 'Maximum 1000 transactions par batch'
            }), 400
        
        # Faire les prédictions
        result = PredictionService.predict_batch(transactions)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/transaction/<int:transaction_id>', methods=['POST'])
@jwt_required()
def predict_existing_transaction(transaction_id):
    """
    Prédire pour une transaction existante en base de données
    et mettre à jour la prédiction
    """
    try:
        # Récupérer la transaction
        transaction = TransactionService.get_transaction_by_id(transaction_id)
        
        if not transaction:
            return jsonify({
                'success': False,
                'error': 'Transaction non trouvée'
            }), 404
        
        # Convertir en dict pour la prédiction
        transaction_data = {
            'gender': transaction.gender,
            'age': transaction.age,
            'house_type_id': transaction.house_type_id,
            'contact_avaliability_id': transaction.contact_avaliability_id,
            'home_country': transaction.home_country,
            'account_no': transaction.account_no,
            'card_expiry_date': transaction.card_expiry_date,
            'transaction_amount': transaction.transaction_amount,
            'transaction_country': transaction.transaction_country,
            'large_purchase': transaction.large_purchase,
            'product_id': transaction.product_id,
            'cif': transaction.cif,
            'transaction_currency_code': transaction.transaction_currency_code
        }
        
        # Faire la prédiction
        result = PredictionService.predict(transaction_data)
        
        if not result['success']:
            return jsonify(result), 500
        
        # Mettre à jour la transaction avec la prédiction
        update_data = {
            'prediction': result['prediction'],
            'prediction_proba': result['probability']['fraud'] / 100
        }
        
        updated_transaction, error = TransactionService.update_transaction(
            transaction_id, 
            update_data
        )
        
        if error:
            return jsonify({
                'success': False,
                'error': f'Prédiction réussie mais erreur lors de la mise à jour : {error}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Prédiction effectuée et transaction mise à jour',
            'prediction': {
                'is_fraud': result['is_fraud'],
                'label': result['label'],
                'probability': result['probability'],
                'confidence': result['confidence']
            },
            'transaction': updated_transaction.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/transactions/pending', methods=['POST'])
@jwt_required()
def predict_all_pending():
    """
    Prédire pour toutes les transactions sans prédiction
    """
    user_id: int = int(get_jwt_identity())
    try:
        from models.transaction import Transaction
        from db.connexion.connexion import db
        
        # Récupérer les transactions sans prédiction
        pending_transactions = Transaction.query.filter(
            Transaction.prediction.is_(None),
            Transaction.user_id == user_id,
        ).limit(100).all()  # Limiter à 100 pour éviter la surcharge
        
        if not pending_transactions:
            return jsonify({
                'success': True,
                'message': 'Aucune transaction en attente de prédiction',
                'processed': 0
            }), 200
        
        results = []
        success_count = 0
        
        for transaction in pending_transactions:
            # Préparer les données
            transaction_data = {
                'gender': transaction.gender,
                'age': transaction.age,
                'house_type_id': transaction.house_type_id,
                'contact_avaliability_id': transaction.contact_avaliability_id,
                'home_country': transaction.home_country,
                'account_no': transaction.account_no,
                'card_expiry_date': transaction.card_expiry_date,
                'transaction_amount': transaction.transaction_amount,
                'transaction_country': transaction.transaction_country,
                'large_purchase': transaction.large_purchase,
                'product_id': transaction.product_id,
                'cif': transaction.cif,
                'transaction_currency_code': transaction.transaction_currency_code
            }
            
            # Faire la prédiction
            result = PredictionService.predict(transaction_data)
            
            if result['success']:
                # Mettre à jour la transaction
                transaction.prediction = result['prediction']
                transaction.prediction_proba = result['probability']['fraud'] / 100
                success_count += 1
                
                results.append({
                    'id': transaction.id,
                    'prediction': result['prediction'],
                    'is_fraud': result['is_fraud']
                })
        
        # Sauvegarder toutes les mises à jour
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'{success_count} transactions traitées',
            'processed': success_count,
            'total_pending': len(pending_transactions),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500