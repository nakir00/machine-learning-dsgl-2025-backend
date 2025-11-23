"""
Service d'authentification - Logique métier pour l'auth JWT
"""
from db.connexion.connexion import db
from models.user_auth import UserAuth
from models.token_blacklist import TokenBlacklist
from flask_jwt_extended import create_access_token, create_refresh_token, get_jwt
from datetime import datetime
from sqlalchemy.exc import IntegrityError


class AuthService:
    """Service pour gérer l'authentification"""
    
    @staticmethod
    def register(email:str, username:str, password:str, first_name:str=None, last_name:str=None)-> UserAuth:
        """
        Enregistre un nouvel utilisateur
        
        Returns:
            tuple: (user, error_message)
        """
        try:
            # Vérifier si l'email existe déjà
            if UserAuth.find_by_email(email):
                return None, "Cet email est déjà utilisé"
            
            # Vérifier si le username existe déjà
            if UserAuth.find_by_username(username):
                return None, "Ce nom d'utilisateur est déjà pris"
            
            # Créer l'utilisateur
            user = UserAuth(
                email=email,
                username=username,
                first_name=first_name,
                last_name=last_name
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            return user, None
            
        except IntegrityError:
            db.session.rollback()
            return None, "Erreur lors de la création du compte"
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur: {str(e)}"
    
    @staticmethod
    def login(email:str, password:str)-> UserAuth:
        """
        Authentifie un utilisateur et génère les tokens
        
        Returns:
            tuple: (tokens_dict, error_message)
        """
        try:
            # Trouver l'utilisateur par email
            user: UserAuth = UserAuth.find_by_email(email)
            
            if not user:
                return None, "Email ou mot de passe incorrect"
            
            # Vérifier si le compte est actif
            if not user.is_active:
                return None, "Ce compte est désactivé"
            
            # Vérifier le mot de passe
            if not user.check_password(password):
                return None, "Email ou mot de passe incorrect"
            
            # Mettre à jour la dernière connexion
            user.update_last_login()
            db.session.commit()
            
            # Générer les tokens (identity doit être une string)
            access_token = create_access_token(identity=str(user.id))
            refresh_token = create_refresh_token(identity=str(user.id))
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'user': user.to_dict()
            }, None
            
        except Exception as e:
            return None, f"Erreur lors de la connexion: {str(e)}"
    
    @staticmethod
    def refresh_token(user_id: int):
        """
        Génère un nouveau access token
        
        Returns:
            tuple: (new_access_token, error_message)
        """
        try:
            user = UserAuth.find_by_id(user_id)
            
            if not user:
                return None, "Utilisateur non trouvé"
            
            if not user.is_active:
                return None, "Ce compte est désactivé"
            
            new_access_token = create_access_token(identity=str(user_id))
            
            return {'access_token': new_access_token}, None
            
        except Exception as e:
            return None, f"Erreur: {str(e)}"
    
    @staticmethod
    def logout(jti: str, token_type: str, user_id: int, expires_at: datetime):
        """
        Déconnecte l'utilisateur en ajoutant le token à la blacklist
        
        Returns:
            tuple: (success, error_message)
        """
        try:
            TokenBlacklist.add_token(
                jti=jti,
                token_type=token_type,
                user_id=user_id,
                expires_at=expires_at
            )
            return True, None
        except Exception as e:
            return False, f"Erreur lors de la déconnexion: {str(e)}"
    
    @staticmethod
    def get_user_profile(user_id: int) -> UserAuth:
        """
        Récupère le profil de l'utilisateur connecté
        
        Returns:
            tuple: (user, error_message)
        """
        try:
            user = UserAuth.find_by_id(user_id)
            
            if not user:
                return None, "Utilisateur non trouvé"
            
            return user, None
            
        except Exception as e:
            return None, f"Erreur: {str(e)}"
    
    @staticmethod
    def change_password(user_id: int, current_password: str, new_password: str):
        """
        Change le mot de passe de l'utilisateur
        
        Returns:
            tuple: (success, error_message)
        """
        print("changing password in service")
        try:
            user: UserAuth = UserAuth.find_by_id(user_id)
            
            if not user:
                return False, "Utilisateur non trouvé"
            
            # Vérifier l'ancien mot de passe
            if not user.check_password(current_password):
                return False, "Mot de passe actuel incorrect"
            
            # Définir le nouveau mot de passe
            user.set_password(new_password)
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            db.session.rollback()
            return False, f"Erreur: {str(e)}"
    
    @staticmethod
    def update_profile(user_id: int, data: dict):
        """
        Met à jour le profil de l'utilisateur
        
        Returns:
            tuple: (user, error_message)
        """
        try:
            user: UserAuth = UserAuth.find_by_id(user_id)
            
            if not user:
                return None, "Utilisateur non trouvé"
            
            # Mettre à jour les champs autorisés
            if 'first_name' in data:
                user.first_name = data['first_name']
            if 'last_name' in data:
                user.last_name = data['last_name']
            if 'username' in data:
                # Vérifier si le nouveau username est disponible
                existing = UserAuth.find_by_username(data['username'])
                if existing and existing.id != user_id:
                    return None, "Ce nom d'utilisateur est déjà pris"
                user.username = data['username']
            
            db.session.commit()
            
            return user, None
            
        except Exception as e:
            db.session.rollback()
            return None, f"Erreur: {str(e)}"
    
    @staticmethod
    def is_token_revoked(jti):
        """
        Vérifie si un token est révoqué
        
        Returns:
            bool: True si le token est révoqué
        """
        return TokenBlacklist.is_token_blacklisted(jti)