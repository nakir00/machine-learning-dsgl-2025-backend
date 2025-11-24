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
        frontend_url = os.environ.get('FRONTEND_URL', '*')
        cors_config = {
            'origins': frontend_url if frontend_url != '*' else '*',
            'supports_credentials': True,
            'allow_headers': [
                'Content-Type',
                'Authorization',
                'Access-Control-Allow-Credentials'
            ],
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH']
        }
    
    CORS(app, **cors_config)
    
    print(f"✅ CORS configuré pour l'environnement: {env}")
    if env == 'local':
        print(f"   Origines autorisées: {cors_config['origins']}")
    else:
        print(f"   Origine autorisée: {frontend_url}")
    
    return app