.PHONY: help install run dev docker-build docker-up docker-down test clean

# Variables
PYTHON := python3
PIP := pip3
FLASK := flask

help: ## Affiche l'aide
	@echo "Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installe les dépendances
	$(PIP) install -r requirements.txt

run: ## Lance l'application
	$(PYTHON) main.py

dev: ## Lance l'application en mode développement
	export ENV=local && $(PYTHON) main.py

docker-build: ## Construit l'image Docker
	docker build -t fraud-detection-api .

docker-up: ## Lance l'application avec Docker Compose
	docker-compose up -d

docker-down: ## Arrête Docker Compose
	docker-compose down

docker-logs: ## Affiche les logs Docker
	docker-compose logs -f app

docker-restart: ## Redémarre les conteneurs
	docker-compose restart

test-api: ## Teste les endpoints de l'API
	@echo "Testing health endpoint..."
	curl http://localhost:8080/health
	@echo "\n\nTesting root endpoint..."
	curl http://localhost:8080/

clean: ## Nettoie les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

deploy: ## Déploie sur Cloud Run
	gcloud run deploy fraud-detection-api \
		--source . \
		--region europe-west1 \
		--allow-unauthenticated

setup-env: ## Crée le fichier .env depuis .env.example
	cp .env.example .env
	@echo "Fichier .env créé. Modifiez-le avec vos paramètres."

# Commandes pour la structure du projet
create-structure: ## Crée la structure de dossiers du projet
	mkdir -p db/connexion
	mkdir -p models
	mkdir -p validators
	mkdir -p services
	mkdir -p routes
	mkdir -p utils
	touch db/__init__.py db/connexion/__init__.py
	touch models/__init__.py
	touch validators/__init__.py
	touch services/__init__.py
	touch routes/__init__.py
	touch utils/__init__.py
	@echo "Structure créée avec succès!"