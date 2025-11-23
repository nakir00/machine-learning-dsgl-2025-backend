"""
Service Transaction - Logique métier pour les transactions
"""
from db.connexion.connexion import db
from models.transaction import Transaction
from sqlalchemy import func, desc


class TransactionService:
    """Service pour gérer la logique métier des transactions"""
    
    @staticmethod
    def get_all_transactions(page:int=1, per_page:int=20):
        """
        Récupère toutes les transactions avec pagination
        
        Args:
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Transactions et informations de pagination
        """
        pagination = Transaction.query.order_by(Transaction.created_at.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return {
            'transactions': [t.to_dict() for t in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }
    
    @staticmethod
    def get_transaction_by_id(transaction_id:int)-> Transaction:
        """
        Récupère une transaction par son ID
        
        Args:
            transaction_id (int): ID de la transaction
            
        Returns:
            Transaction: Objet Transaction ou None
        """
        return Transaction.query.get(transaction_id)
    
    @staticmethod
    def create_transaction(data:dict):
        """
        Crée une nouvelle transaction
        
        Args:
            data (dict): Données de la transaction (doit inclure user_id)
            
        Returns:
            tuple: (Transaction ou None, error message ou None)
        """
        try:
            # Vérifier que user_id est présent
            if not data.get('user_id'):
                return None, "user_id est requis"
            
            # Créer la transaction depuis le dict
            new_transaction: Transaction = Transaction.from_dict(data)
            
            # Sauvegarder en base
            db.session.add(new_transaction)
            db.session.commit()
            
            return new_transaction, None
            
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur lors de la création : {str(e)}"
    
    @staticmethod
    def update_transaction(transaction_id:int, data:dict):
        """
        Met à jour une transaction
        
        Args:
            transaction_id (int): ID de la transaction
            data (dict): Nouvelles données
            
        Returns:
            tuple: (Transaction ou None, error message ou None)
        """
        try:
            transaction: Transaction = Transaction.query.get(transaction_id)
            
            if not transaction:
                return None, "Transaction non trouvée"
            
            # Mettre à jour les champs fournis
            updatable_fields = [
                'gender', 'age', 'house_type_id', 'contact_avaliability_id',
                'home_country', 'account_no', 'card_expiry_date', 'cif',
                'transaction_amount', 'transaction_country', 'transaction_currency_code',
                'large_purchase', 'product_id', 'potential_fraud', 'prediction',
                'prediction_proba'
            ]
            
            for field in updatable_fields:
                if field in data:
                    setattr(transaction, field, data[field])
            
            db.session.commit()
            
            return transaction, None
            
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur lors de la mise à jour : {str(e)}"
    
    @staticmethod
    def delete_transaction(transaction_id:int):
        """
        Supprime une transaction
        
        Args:
            transaction_id (int): ID de la transaction
            
        Returns:
            tuple: (success: bool, error message ou None)
        """
        try:
            transaction: Transaction = Transaction.query.get(transaction_id)
            
            if not transaction:
                return False, "Transaction non trouvée"
            
            db.session.delete(transaction)
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            db.session.rollback()
            return False, f"Erreur lors de la suppression : {str(e)}"
    
    @staticmethod
    def get_fraud_transactions(page:int=1, per_page:int=20):
        """
        Récupère uniquement les transactions frauduleuses
        
        Args:
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Transactions frauduleuses et pagination
        """
        pagination = Transaction.query.filter_by(potential_fraud=1).order_by(
            Transaction.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return {
            'transactions': [t.to_dict() for t in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }
    
    @staticmethod
    def get_transactions_by_account(account_no:int, page:int=1, per_page:int=20):
        """
        Récupère les transactions d'un compte spécifique
        
        Args:
            account_no (int): Numéro de compte
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Transactions et pagination
        """
        pagination = Transaction.query.filter_by(account_no=account_no).order_by(
            Transaction.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return {
            'transactions': [t.to_dict() for t in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages
            }
        }
    
    @staticmethod
    def get_transactions_by_user(user_id:int, page:int=1, per_page:int=20):
        """
        Récupère les transactions d'un utilisateur spécifique
        
        Args:
            user_id (int): ID de l'utilisateur
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Transactions et pagination
        """
        pagination = Transaction.query.filter_by(user_id=user_id).order_by(
            Transaction.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return {
            'transactions': [t.to_dict() for t in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }
    
    @staticmethod
    def get_user_transaction_statistics(user_id:int):
        """
        Calcule des statistiques sur les transactions d'un utilisateur
        
        Args:
            user_id (int): ID de l'utilisateur
            
        Returns:
            dict: Statistiques diverses
        """
        base_query = Transaction.query.filter_by(user_id=user_id)
        
        total = base_query.count()
        frauds = base_query.filter_by(potential_fraud=1).count()
        legitimate = total - frauds
        
        # Montant total et moyen
        amounts = db.session.query(
            func.sum(Transaction.transaction_amount).label('total_amount'),
            func.avg(Transaction.transaction_amount).label('avg_amount')
        ).filter(Transaction.user_id == user_id).first()
        
        # Montant max
        max_transaction = base_query.order_by(
            desc(Transaction.transaction_amount)
        ).first()
        
        return {
            'total_transactions': total,
            'fraudulent': frauds,
            'legitimate': legitimate,
            'fraud_rate': round((frauds / total * 100), 2) if total > 0 else 0,
            'total_amount': float(amounts.total_amount) if amounts.total_amount else 0,
            'average_amount': round(float(amounts.avg_amount), 2) if amounts.avg_amount else 0,
            'max_amount': float(max_transaction.transaction_amount) if max_transaction else 0
        }
    
    @staticmethod
    def get_transaction_statistics():
        """
        Calcule des statistiques sur les transactions
        
        Returns:
            dict: Statistiques diverses
        """
        total: int = Transaction.query.count()
        frauds: int = Transaction.query.filter_by(potential_fraud=1).count()
        legitimate: int = total - frauds
        
        # Montant total et moyen
        amounts = db.session.query(
            func.sum(Transaction.transaction_amount).label('total_amount'),
            func.avg(Transaction.transaction_amount).label('avg_amount')
        ).first()
        
        # Montant max
        max_transaction = Transaction.query.order_by(
            desc(Transaction.transaction_amount)
        ).first()
        
        return {
            'total_transactions': total,
            'fraudulent': frauds,
            'legitimate': legitimate,
            'fraud_rate': round((frauds / total * 100), 2) if total > 0 else 0,
            'total_amount': float(amounts.total_amount) if amounts.total_amount else 0,
            'average_amount': float(amounts.avg_amount) if amounts.avg_amount else 0,
            'max_amount': float(max_transaction.transaction_amount) if max_transaction else 0
        }
    
    @staticmethod
    def search_transactions_by_amount(min_amount:float=None, max_amount:float=None, page:int=1, per_page:int=20):
        """
        Recherche des transactions par plage de montant
        
        Args:
            min_amount (float): Montant minimum
            max_amount (float): Montant maximum
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Transactions et pagination
        """
        query = Transaction.query
        
        if min_amount is not None:
            query = query.filter(Transaction.transaction_amount >= min_amount)
        
        if max_amount is not None:
            query = query.filter(Transaction.transaction_amount <= max_amount)
        
        pagination = query.order_by(Transaction.created_at.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return {
            'transactions': [t.to_dict() for t in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages
            }
        }
    
    @staticmethod
    def mark_as_fraud(transaction_id:int, is_fraud:bool=True):
        """
        Marque une transaction comme frauduleuse ou légitime
        
        Args:
            transaction_id (int): ID de la transaction
            is_fraud (bool): True pour fraude, False pour légitime
            
        Returns:
            tuple: (Transaction ou None, error message ou None)
        """
        try:
            transaction: Transaction = Transaction.query.get(transaction_id)
            
            if not transaction:
                return None, "Transaction non trouvée"
            
            transaction.potential_fraud = 1 if is_fraud else 0
            db.session.commit()
            
            return transaction, None
            
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur lors de la mise à jour : {str(e)}"