# 🏦 API de Détection de Fraude Bancaire

API REST Flask avec architecture modulaire pour la détection et la gestion des transactions frauduleuses.

## 📁 Structure du projet

```
fraud-detection-api/
├── main.py                          # Point d'entrée
├── requirements.txt                 # Dépendances
├── Dockerfile                       # Configuration Docker
├── docker-compose.yml              # Docker Compose
├── Makefile                        # Commandes pratiques
├── .env.example                    # Exemple de configuration
├── .env                            # Configuration locale (ignoré)
│
├── db/                             # Base de données
│   ├── __init__.py
│   └── connexion/
│       ├── __init__.py
│       └── connexion.py            # Configuration SQLAlchemy
│
├── models/                         # Modèles de données (ORM)
│   ├── __init__.py
│   ├── user.py                     # Modèle User
│   └── transaction.py              # Modèle Transaction
│
├── validators/                     # Validation avec Flask-WTF
│   ├── __init__.py
│   ├── user_validator.py           # Validation User
│   └── transaction_validator.py    # Validation Transaction
│
├── services/                       # Logique métier
│   ├── __init__.py
│   ├── user_service.py             # Service User
│   └── transaction_service.py      # Service Transaction
│
└── routes/                         # Routes API
    ├── __init__.py
    ├── user_routes.py              # Endpoints User
    └── transaction_routes.py       # Endpoints Transaction
```

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd fraud-detection-api

# Installer les dépendances
make install
# ou
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Créer le fichier .env
make setup-env
# ou
cp .env.example .env

# Éditer .env avec vos paramètres MySQL
nano .env
```

### 3. Lancer l'application

#### Option A : Local avec Python

```bash
# Mode développement
make dev

# Mode production
make run
```

#### Option B : Avec Docker Compose (Recommandé)

```bash
# Lancer MySQL + API
make docker-up

# Voir les logs
make docker-logs

# Arrêter
make docker-down
```

### 4. Tester l'API

```bash
# Health check
curl http://localhost:8080/health

# Documentation
curl http://localhost:8080/

# Créer un utilisateur
curl -X POST http://localhost:8080/users \
  -H "Content-Type: application/json" \
  -d '{"username":"john","email":"john@example.com"}'
```

## 📊 Architecture

### Flux de données

```
Request → Route → Validator → Service → Model → Database
            ↓         ↓           ↓
          JSON   Validation   Business
                              Logic
```

### Séparation des responsabilités

| Couche | Responsabilité | Exemple |
|--------|---------------|---------|
| **Routes** | Gestion HTTP, parsing requêtes | `user_routes.py` |
| **Validators** | Validation données avec Flask-WTF | `user_validator.py` |
| **Services** | Logique métier | `user_service.py` |
| **Models** | Représentation BDD (ORM) | `user.py` |
| **DB** | Connexion base de données | `connexion.py` |

## 🔌 Endpoints API

### 🏠 Général

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Documentation API |
| `GET` | `/health` | Statut de l'API |

### 👤 Utilisateurs

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/users` | Liste des utilisateurs (pagination) |
| `GET` | `/users/<id>` | Détails d'un utilisateur |
| `POST` | `/users` | Créer un utilisateur |
| `PUT` | `/users/<id>` | Mettre à jour |
| `DELETE` | `/users/<id>` | Supprimer |
| `GET` | `/users/search?q=` | Rechercher |
| `GET` | `/users/stats` | Statistiques |

### 💳 Transactions

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/transactions` | Liste (pagination) |
| `GET` | `/transactions/<id>` | Détails |
| `POST` | `/transactions` | Créer |
| `PUT` | `/transactions/<id>` | Mettre à jour |
| `DELETE` | `/transactions/<id>` | Supprimer |
| `GET` | `/transactions/fraud` | Fraudes uniquement |
| `GET` | `/transactions/account/<no>` | Par compte |
| `GET` | `/transactions/stats` | Statistiques |
| `GET` | `/transactions/search` | Par montant |
| `POST` | `/transactions/<id>/mark-fraud` | Marquer fraude |

## 📝 Exemples d'utilisation

### Créer un utilisateur

```bash
curl -X POST http://localhost:8080/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "first_name": "John",
    "last_name": "Doe"
  }'
```

**Réponse :**
```json
{
  "success": true,
  "message": "Utilisateur créé avec succès",
  "data": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "is_active": true,
    "created_at": "2025-01-15T10:30:00"
  }
}
```

### Créer une transaction

```bash
curl -X POST http://localhost:8080/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 0,
    "age": 35,
    "account_no": 1234567,
    "transaction_amount": 150.50,
    "transaction_country": 1,
    "potential_fraud": 0
  }'
```

### Rechercher des utilisateurs

```bash
curl "http://localhost:8080/users/search?q=john&page=1&per_page=10"
```

### Obtenir les statistiques

```bash
# Statistiques utilisateurs
curl http://localhost:8080/users/stats

# Statistiques transactions
curl http://localhost:8080/transactions/stats
```

**Réponse :**
```json
{
  "success": true,
  "stats": {
    "total_transactions": 1000,
    "fraudulent": 25,
    "legitimate": 975,
    "fraud_rate": 2.5,
    "total_amount": 150000.00,
    "average_amount": 150.00,
    "max_amount": 5000.00
  }
}
```

## 🔒 Validation des données

### Validation avec Flask-WTF

Les validateurs utilisent Flask-WTF pour une validation robuste :

```python
# Exemple d'utilisation dans validators/user_validator.py
class CreateUserForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(message="Le nom d'utilisateur est requis"),
        Length(min=3, max=80)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message="Format d'email invalide")
    ])
```

### Validation manuelle (helper)

Pour les cas où Flask-WTF n'est pas nécessaire :

```python
from validators.user_validator import validate_user_data

is_valid, errors = validate_user_data(data, is_update=False)
if not is_valid:
    return {'errors': errors}, 400
```

## 🗄️ Modèles de données

### User

```python
{
    "id": int,
    "username": str (unique, 3-80 caractères),
    "email": str (unique, format email),
    "first_name": str (optionnel),
    "last_name": str (optionnel),
    "is_active": bool,
    "created_at": datetime,
    "updated_at": datetime
}
```

### Transaction

```python
{
    "id": int,
    "gender": int (0 ou 1),
    "age": int (0-150),
    "account_no": int,
    "transaction_amount": float (requis),
    "potential_fraud": int (0 ou 1),
    "prediction": int (0 ou 1),
    "prediction_proba": float (0-1),
    "created_at": datetime,
    "updated_at": datetime
}
```

## 🐳 Docker

### Construction

```bash
# Construire l'image
make docker-build

# Lancer avec Docker Compose
make docker-up
```

### Variables d'environnement

Dans `docker-compose.yml` ou `.env` :

```yaml
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=fraud_user
MYSQL_PASSWORD=secure_password
MYSQL_DATABASE=fraud_detection
```

## 🚀 Déploiement Cloud Run

```bash
# Avec Makefile
make deploy

# Ou directement
gcloud run deploy fraud-detection-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars "MYSQL_HOST=your_host,MYSQL_PORT=3306,MYSQL_USER=user,MYSQL_PASSWORD=pass,MYSQL_DATABASE=fraud_detection"
```

## 🧪 Tests

```bash
# Tester tous les endpoints
make test-api

# Tests manuels
curl http://localhost:8080/health
curl http://localhost:8080/users
curl http://localhost:8080/transactions/stats
```

## 🛠️ Développement

### Ajouter un nouveau modèle

1. Créer `models/nouveau_modele.py`
2. Définir la classe avec SQLAlchemy
3. Ajouter dans `models/__init__.py`
4. Créer le service correspondant
5. Créer le validator
6. Créer les routes

### Ajouter un endpoint

1. Modifier `routes/[entity]_routes.py`
2. Ajouter la logique dans `services/[entity]_service.py`
3. Tester avec `curl`

## 📚 Commandes Makefile

```bash
make help              # Affiche l'aide
make install           # Installe les dépendances
make run               # Lance l'app
make dev               # Lance en mode dev
make docker-build      # Build Docker
make docker-up         # Lance Docker Compose
make docker-down       # Arrête Docker
make docker-logs       # Affiche les logs
make test-api          # Test les endpoints
make clean             # Nettoie les fichiers temp
make deploy            # Déploie sur Cloud Run
make setup-env         # Crée le .env
make create-structure  # Crée la structure
```

## 🔐 Sécurité

- ✅ Validation des données avec Flask-WTF
- ✅ Protection contre les injections SQL (SQLAlchemy ORM)
- ✅ Variables d'environnement pour les secrets
- ✅ Gestion des erreurs centralisée
- ✅ CSRF désactivé (API REST sans sessions)

## 📈 Performances

- Pool de connexions MySQL configuré
- Pagination sur tous les endpoints de liste
- Limit de 100 résultats max par page
- Index sur les champs fréquemment recherchés

## 🐛 Dépannage

### Erreur de connexion MySQL

```bash
# Vérifier les variables d'environnement
echo $MYSQL_HOST

# Tester la connexion
mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE

# Vérifier les logs
make docker-logs
```

### Tables non créées

Les tables sont créées automatiquement au démarrage. Si problème :

```python
# Dans un shell Python
from main import app, db
with app.app_context():
    db.create_all()
```

## 📄 Licence

MIT

## 👥 Contributeurs

Votre équipe ici

---

**🎯 Projet prêt pour la production !**