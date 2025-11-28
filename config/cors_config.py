"""
Configuration CORS pour l'application Flask
"""
import os

def init_cors(app):
    """Configure CORS selon l'environnement"""
    from flask_cors import CORS
    
    env = os.environ.get('ENV', 'production')
    
    if env == 'local':
        # Développement local
        cors_config = {
            'origins': "*",
            'supports_credentials': True,
            'allow_headers': [
                'Content-Type',
                'Authorization',
                'Access-Control-Allow-Credentials'
            ],
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
            'expose_headers': ['Content-Type', 'Authorization']
        }
    else:
        # Production
        raw_frontend_url = os.environ.get('FRONTEND_URL', '*').strip()

        if raw_frontend_url == '*' or raw_frontend_url == '':
            origins = '*'
        else:
            # Sépare par virgule et nettoie chaque URL
            """ origins = [
                origin.strip() 
                for origin in raw_frontend_url.split(',')
                if origin.strip()  # ignore les vides
            ] """

        cors_config = {
            'origins': '*',
            'supports_credentials': True,
            'allow_headers': [
                'Content-Type',
                'Authorization',
                'Access-Control-Allow-Credentials',
                'X-Requested-With',  # utile pour certains clients
            ],
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
            'expose_headers': ['Content-Type', 'Authorization'],  # optionnel mais propre
        }
    
    CORS(app, **cors_config)
    
    print(f"✅ CORS configuré pour l'environnement: {env}")
    if env == 'local':
        print(f"   Origines autorisées: {cors_config['origins']}")
    else:
        print(f"   Origine autorisée....")
    
    return app