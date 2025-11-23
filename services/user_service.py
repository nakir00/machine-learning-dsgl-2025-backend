"""
Service User - Logique métier pour les utilisateurs
"""
from db.connexion.connexion import db
from models.user import User
from sqlalchemy.exc import IntegrityError


class UserService:
    """Service pour gérer la logique métier des utilisateurs"""
    
    @staticmethod
    def get_all_users(page: int = 1, per_page: int = 20):
        """
        Récupère tous les utilisateurs avec pagination
        
        Args:
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Utilisateurs et informations de pagination
        """
        pagination = User.query.order_by(User.created_at.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return {
            'users': [user.to_dict() for user in pagination.items],
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
    def get_user_by_id(user_id: int):
        """
        Récupère un utilisateur par son ID
        
        Args:
            user_id (int): ID de l'utilisateur
            
        Returns:
            User: Objet User ou None
        """
        return User.query.get(user_id)
    
    @staticmethod
    def get_user_by_username(username: str):
        """
        Récupère un utilisateur par son username
        
        Args:
            username (str): Username de l'utilisateur
            
        Returns:
            User: Objet User ou None
        """
        return User.query.filter_by(username=username).first()
    
    @staticmethod
    def get_user_by_email(email: str):
        """
        Récupère un utilisateur par son email
        
        Args:
            email (str): Email de l'utilisateur
            
        Returns:
            User: Objet User ou None
        """
        return User.query.filter_by(email=email).first()
    
    @staticmethod
    def create_user(data: dict):
        """
        Crée un nouvel utilisateur
        
        Args:
            data (dict): Données de l'utilisateur
            
        Returns:
            tuple: (User ou None, error message ou None)
        """
        try:
            # Créer l'utilisateur depuis le dict
            new_user = User.from_dict(data)
            
            # Sauvegarder en base
            db.session.add(new_user)
            db.session.commit()
            
            return new_user, None
            
        except IntegrityError as e:
            db.session.rollback()
            
            # Gérer les erreurs de contrainte unique
            if 'username' in str(e.orig):
                return None, "Ce nom d'utilisateur existe déjà"
            elif 'email' in str(e.orig):
                return None, "Cet email est déjà utilisé"
            else:
                return None, f"Erreur d'intégrité : {str(e)}"
        
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur lors de la création : {str(e)}"
    
    @staticmethod
    def update_user(user_id: int, data: dict):
        """
        Met à jour un utilisateur
        
        Args:
            user_id (int): ID de l'utilisateur
            data (dict): Nouvelles données
            
        Returns:
            tuple: (User ou None, error message ou None)
        """
        try:
            user: User = User.query.get(user_id)
            
            if not user:
                return None, "Utilisateur non trouvé"
            
            # Mettre à jour les champs fournis
            if 'username' in data:
                user.username = data['username']
            if 'email' in data:
                user.email = data['email']
            if 'first_name' in data:
                user.first_name = data['first_name']
            if 'last_name' in data:
                user.last_name = data['last_name']
            if 'is_active' in data:
                user.is_active = data['is_active']
            
            db.session.commit()
            
            return user, None
            
        except IntegrityError as e:
            db.session.rollback()
            
            if 'username' in str(e.orig):
                return None, "Ce nom d'utilisateur existe déjà"
            elif 'email' in str(e.orig):
                return None, "Cet email est déjà utilisé"
            else:
                return None, f"Erreur d'intégrité : {str(e)}"
        
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur lors de la mise à jour : {str(e)}"
    
    @staticmethod
    def delete_user(user_id: int):
        """
        Supprime un utilisateur
        
        Args:
            user_id (int): ID de l'utilisateur
            
        Returns:
            tuple: (success: bool, error message ou None)
        """
        try:
            user = User.query.get(user_id)
            
            if not user:
                return False, "Utilisateur non trouvé"
            
            db.session.delete(user)
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            db.session.rollback()
            return False, f"Erreur lors de la suppression : {str(e)}"
    
    @staticmethod
    def search_users(query: str, page: int = 1, per_page: int = 20):
        """
        Recherche des utilisateurs par username ou email
        
        Args:
            query (str): Terme de recherche
            page (int): Numéro de la page
            per_page (int): Nombre d'éléments par page
            
        Returns:
            dict: Résultats de recherche et pagination
        """
        search = f"%{query}%"
        
        pagination = User.query.filter(
            (User.username.like(search)) | (User.email.like(search))
        ).order_by(User.created_at.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return {
            'users': [user.to_dict() for user in pagination.items],
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
    def get_active_users_count():
        """
        Compte le nombre d'utilisateurs actifs
        
        Returns:
            int: Nombre d'utilisateurs actifs
        """
        return User.query.filter_by(is_active=True).count()
    
    @staticmethod
    def get_total_users_count():
        """
        Compte le nombre total d'utilisateurs
        
        Returns:
            int: Nombre total d'utilisateurs
        """
        return User.query.count()