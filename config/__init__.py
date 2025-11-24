"""Package de configuration"""
from config.jwt_config import init_jwt
from config.cors_config import init_cors

__all__ = ['init_jwt', 'init_cors']