# Utilise une image légère Python
FROM python:3.11-slim

# Définit le dossier de travail dans le container
WORKDIR /app

# Copie les dépendances
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code dans l'image
COPY . .

# Définit une variable d’environnement pour Flask (optionnel si app.py contient app.run())
ENV PORT=8080

# Expose le port
EXPOSE 8080

# Commande pour lancer l’application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app
