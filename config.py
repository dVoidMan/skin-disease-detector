import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    """Base Configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-insecure-secret-key-change-in-production')
    
    # Database Settings (Default: SQLite in database/ directory)
    raw_db_url = os.environ.get('DATABASE_URL')
    if not raw_db_url or raw_db_url == 'sqlite:///database/skin_disease.db':
        # Resolve to absolute path with forward slashes for cross-platform and docker safety
        db_default_path = os.path.join(basedir, 'database', 'skin_disease.db').replace('\\', '/')
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_default_path}"
    elif raw_db_url.startswith('sqlite:///') and not raw_db_url.startswith('sqlite:////') and not (len(raw_db_url) > 11 and raw_db_url[11] == ':'):
        # Convert relative SQLite URI to absolute path
        rel_path = raw_db_url[len('sqlite:///'):]
        abs_path = os.path.abspath(os.path.join(basedir, rel_path)).replace('\\', '/')
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{abs_path}"
    else:
        SQLALCHEMY_DATABASE_URI = raw_db_url

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload Settings
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16 MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    
    # Cloudflare R2 Object Storage (S3-Compatible)
    CLOUDFLARE_R2_ENABLED = os.environ.get('CLOUDFLARE_R2_ENABLED', 'false').lower() in ('true', '1', 'yes', 'on')
    R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID', '')
    R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID', '')
    R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY', '')
    R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'skin-disease-images')
    R2_PUBLIC_URL = os.environ.get('R2_PUBLIC_URL', '').rstrip('/')
    
    @property
    def R2_ENDPOINT_URL(self):
        if self.R2_ACCOUNT_ID:
            return f"https://{self.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        return None

class DevelopmentConfig(Config):
    """Development Configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production Configuration."""
    DEBUG = False

class TestingConfig(Config):
    """Testing Configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
