"""
Service de Prédiction - Détection de fraude avec Machine Learning
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionService:
    """Service pour la prédiction de fraude avec le modèle ML"""
    
    # Chemins par défaut des fichiers du modèle
    DEFAULT_MODEL_PATH = 'ml/models/model.pkl'
    DEFAULT_SCALER_PATH = 'ml/models/scaler.pkl'
    DEFAULT_STATS_PATH = 'ml/models/train_stats.pkl'
    
    # Instance singleton
    _instance = None
    _model = None
    _scaler = None
    _train_stats = None
    _is_loaded = False
    
    def __new__(cls):
        """Singleton pattern pour éviter de recharger le modèle"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le service et charge le modèle si pas déjà fait"""
        if not PredictionService._is_loaded:
            self.load_model()
    
    @classmethod
    def load_model(cls, model_path=None, scaler_path=None, stats_path=None):
        """
        Charge le modèle ML, le scaler et les statistiques d'entraînement
        
        Args:
            model_path (str): Chemin vers le modèle .pkl
            scaler_path (str): Chemin vers le scaler .pkl
            stats_path (str): Chemin vers les stats d'entraînement .pkl
            
        Returns:
            bool: True si chargement réussi, False sinon
        """
        model_path = model_path or os.environ.get('MODEL_PATH', cls.DEFAULT_MODEL_PATH)
        scaler_path = scaler_path or os.environ.get('SCALER_PATH', cls.DEFAULT_SCALER_PATH)
        stats_path = stats_path or os.environ.get('STATS_PATH', cls.DEFAULT_STATS_PATH)
        
        try:
            # Charger le modèle
            if Path(model_path).exists():
                cls._model = joblib.load(model_path)
                print(f"✅ Modèle chargé : {model_path}")
            else:
                print(f"⚠️ Modèle non trouvé : {model_path}")
                return False
            
            # Charger le scaler
            if Path(scaler_path).exists():
                cls._scaler = joblib.load(scaler_path)
                print(f"✅ Scaler chargé : {scaler_path}")
            else:
                print(f"⚠️ Scaler non trouvé : {scaler_path}")
                return False
            
            # Charger les statistiques (optionnel)
            if Path(stats_path).exists():
                cls._train_stats = joblib.load(stats_path)
                print(f"✅ Statistiques chargées : {stats_path}")
            else:
                print(f"ℹ️ Statistiques non trouvées (optionnel) : {stats_path}")
                cls._train_stats = None
            
            cls._is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            cls._is_loaded = False
            return False
    
    @classmethod
    def is_model_loaded(cls):
        """
        Vérifie si le modèle est chargé
        
        Returns:
            bool: True si le modèle est prêt
        """
        return cls._is_loaded and cls._model is not None and cls._scaler is not None
    
    @classmethod
    def get_model_info(cls):
        """
        Retourne les informations sur le modèle chargé
        
        Returns:
            dict: Informations sur le modèle
        """
        if not cls._is_loaded:
            return {
                'loaded': False,
                'message': 'Modèle non chargé'
            }
        
        return {
            'loaded': True,
            'model_type': type(cls._model).__name__,
            'has_scaler': cls._scaler is not None,
            'has_train_stats': cls._train_stats is not None
        }
    
    @staticmethod
    def preprocess_transaction(transaction_data, train_stats=None):
        """
        Applique le même preprocessing que lors de l'entraînement
        
        Args:
            transaction_data (dict): Données de la transaction
            train_stats (dict): Statistiques d'entraînement (optionnel)
            
        Returns:
            pd.DataFrame: Données preprocessées
        """
        # Convertir en DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Utiliser les stats d'entraînement si disponibles
        if train_stats is None:
            train_stats = PredictionService._train_stats
        
        # 1. Créer l'indicateur de transaction élevée
        if train_stats and 'percentile_99' in train_stats:
            percentile_99 = train_stats['percentile_99']
        else:
            percentile_99 = 500  # Valeur par défaut
        
        df['TransactionAmount_is_high'] = (df['TransactionAmount'] > percentile_99).astype(int)
        
        # 2. Calculer le Z-score
        if train_stats and 'mean' in train_stats and 'std' in train_stats:
            mean = train_stats['mean']
            std = train_stats['std']
        else:
            mean = 100  # Valeur par défaut
            std = 200   # Valeur par défaut
        
        # Éviter la division par zéro
        if std == 0:
            std = 1
        
        df['TransactionAmount_zscore'] = (df['TransactionAmount'] - mean) / std
        
        # 3. Transformation logarithmique
        df['TransactionAmount_log'] = np.log1p(df['TransactionAmount'])
        
        # 4. Catégorisation du montant
        bins = [0, 10, 50, 100, 500, float('inf')]
        df['TransactionAmount_category'] = pd.cut(
            df['TransactionAmount'],
            bins=bins,
            labels=False
        ).fillna(0).astype(int)
        
        return df
    
    @staticmethod
    def prepare_features(transaction_data):
        """
        Prépare les features pour la prédiction
        
        Args:
            transaction_data (dict): Données brutes de la transaction
            
        Returns:
            pd.DataFrame: Features prêtes pour le modèle
        """
        # Mapping des noms de colonnes (de l'API vers le modèle)
        column_mapping = {
            'gender': 'Gender',
            'age': 'Age',
            'house_type_id': 'HouseTypeID',
            'contact_avaliability_id': 'ContactAvaliabilityID',
            'home_country': 'HomeCountry',
            'account_no': 'AccountNo',
            'card_expiry_date': 'CardExpiryDate',
            'transaction_amount': 'TransactionAmount',
            'transaction_country': 'TransactionCountry',
            'large_purchase': 'LargePurchase',
            'product_id': 'ProductID',
            'cif': 'CIF',
            'transaction_currency_code': 'TransactionCurrencyCode'
        }
        
        # Convertir les noms de colonnes
        mapped_data = {}
        for api_name, model_name in column_mapping.items():
            if api_name in transaction_data:
                mapped_data[model_name] = transaction_data[api_name]
            elif model_name in transaction_data:
                mapped_data[model_name] = transaction_data[model_name]
            else:
                # Valeur par défaut
                mapped_data[model_name] = 0
        
        # Créer le DataFrame
        df = pd.DataFrame([mapped_data])
        
        # Appliquer le preprocessing
        df = PredictionService.preprocess_transaction(df)
        
        return df
    
    @classmethod
    def predict(cls, transaction_data):
        """
        Prédit si une transaction est frauduleuse
        
        Args:
            transaction_data (dict): Données de la transaction
            
        Returns:
            dict: Résultat de la prédiction avec probabilité
        """
        # Vérifier que le modèle est chargé
        if not cls.is_model_loaded():
            return {
                'success': False,
                'error': 'Modèle non chargé. Veuillez charger le modèle d\'abord.',
                'prediction': None,
                'probability': None
            }
        
        try:
            # Préparer les features
            features_df = cls.prepare_features(transaction_data)
            
            # Normaliser avec le scaler
            features_scaled = cls._scaler.transform(features_df)
            
            # Faire la prédiction
            prediction = cls._model.predict(features_scaled)[0]
            
            # Obtenir les probabilités si disponibles
            if hasattr(cls._model, 'predict_proba'):
                probabilities = cls._model.predict_proba(features_scaled)[0]
                proba_fraud = float(probabilities[1])
                proba_legitimate = float(probabilities[0])
            else:
                proba_fraud = float(prediction)
                proba_legitimate = 1 - proba_fraud
            
            return {
                'success': True,
                'prediction': int(prediction),
                'is_fraud': bool(prediction == 1),
                'label': 'FRAUDE' if prediction == 1 else 'LÉGITIME',
                'probability': {
                    'fraud': round(proba_fraud * 100, 2),
                    'legitimate': round(proba_legitimate * 100, 2)
                },
                'confidence': round(max(proba_fraud, proba_legitimate) * 100, 2)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur lors de la prédiction : {str(e)}',
                'prediction': None,
                'probability': None
            }
    
    @classmethod
    def predict_batch(cls, transactions_list):
        """
        Prédit pour un batch de transactions
        
        Args:
            transactions_list (list): Liste de dictionnaires de transactions
            
        Returns:
            dict: Résultats des prédictions
        """
        if not cls.is_model_loaded():
            return {
                'success': False,
                'error': 'Modèle non chargé',
                'predictions': []
            }
        
        try:
            results = []
            fraud_count = 0
            
            for idx, transaction in enumerate(transactions_list):
                result = cls.predict(transaction)
                result['index'] = idx
                results.append(result)
                
                if result.get('is_fraud'):
                    fraud_count += 1
            
            return {
                'success': True,
                'total': len(transactions_list),
                'fraud_detected': fraud_count,
                'legitimate': len(transactions_list) - fraud_count,
                'fraud_rate': round(fraud_count / len(transactions_list) * 100, 2) if transactions_list else 0,
                'predictions': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur lors de la prédiction batch : {str(e)}',
                'predictions': []
            }
    
    @classmethod
    def explain_prediction(cls, transaction_data):
        """
        Explique les facteurs de risque d'une transaction
        
        Args:
            transaction_data (dict): Données de la transaction
            
        Returns:
            dict: Analyse des facteurs de risque
        """
        risk_factors = []
        risk_score = 0
        
        # Analyser le montant
        amount = transaction_data.get('transaction_amount') or transaction_data.get('TransactionAmount', 0)
        if amount > 500:
            risk_factors.append({
                'factor': 'Montant élevé',
                'value': amount,
                'risk': 'HIGH',
                'description': f'Transaction de {amount}€ supérieure au seuil de 500€'
            })
            risk_score += 30
        elif amount > 200:
            risk_factors.append({
                'factor': 'Montant modéré-élevé',
                'value': amount,
                'risk': 'MEDIUM',
                'description': f'Transaction de {amount}€ dans la tranche supérieure'
            })
            risk_score += 15
        
        # Analyser l'âge
        age = transaction_data.get('age') or transaction_data.get('Age', 30)
        if age < 18:
            risk_factors.append({
                'factor': 'Âge suspect',
                'value': age,
                'risk': 'HIGH',
                'description': 'Client mineur - vérification requise'
            })
            risk_score += 25
        elif age > 80:
            risk_factors.append({
                'factor': 'Client senior',
                'value': age,
                'risk': 'MEDIUM',
                'description': 'Vigilance accrue pour les clients seniors'
            })
            risk_score += 10
        
        # Analyser l'achat large
        large_purchase = transaction_data.get('large_purchase') or transaction_data.get('LargePurchase', 0)
        if large_purchase == 1:
            risk_factors.append({
                'factor': 'Achat important',
                'value': 'Oui',
                'risk': 'MEDIUM',
                'description': 'Transaction marquée comme achat important'
            })
            risk_score += 15
        
        # Analyser le pays de transaction
        home_country = transaction_data.get('home_country') or transaction_data.get('HomeCountry', 1)
        trans_country = transaction_data.get('transaction_country') or transaction_data.get('TransactionCountry', 1)
        if home_country != trans_country:
            risk_factors.append({
                'factor': 'Transaction internationale',
                'value': f'Pays domicile: {home_country}, Pays transaction: {trans_country}',
                'risk': 'MEDIUM',
                'description': 'Transaction effectuée dans un pays différent du domicile'
            })
            risk_score += 20
        
        # Déterminer le niveau de risque global
        if risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level,
            'factors_count': len(risk_factors),
            'risk_factors': risk_factors,
            'recommendation': 'Vérification manuelle recommandée' if risk_level == 'HIGH' else 'Aucune action requise'
        }