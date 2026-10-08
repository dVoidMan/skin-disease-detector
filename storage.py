import os
import uuid
import logging
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Unified Storage Manager for Cloudflare R2 and Local Disk Storage.
    
    If Cloudflare R2 credentials are provided and CLOUDFLARE_R2_ENABLED is True,
    files are uploaded to Cloudflare R2 via its S3-compatible API.
    Otherwise, files seamlessly fall back to local disk storage in static/uploads/.
    """
    def __init__(self, app=None):
        self.r2_client = None
        self.r2_enabled = False
        self.bucket_name = None
        self.public_url = None
        self.local_upload_folder = None
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        
        if app:
            self.init_app(app)

    def init_app(self, app):
        self.local_upload_folder = app.config.get('UPLOAD_FOLDER', os.path.join(app.root_path, 'static', 'uploads'))
        os.makedirs(self.local_upload_folder, exist_ok=True)
        
        self.allowed_extensions = app.config.get('ALLOWED_EXTENSIONS', self.allowed_extensions)
        self.r2_enabled = app.config.get('CLOUDFLARE_R2_ENABLED', False)
        self.bucket_name = app.config.get('R2_BUCKET_NAME')
        self.public_url = app.config.get('R2_PUBLIC_URL', '')
        
        account_id = app.config.get('R2_ACCOUNT_ID')
        access_key = app.config.get('R2_ACCESS_KEY_ID')
        secret_key = app.config.get('R2_SECRET_ACCESS_KEY')
        
        if self.r2_enabled and account_id and access_key and secret_key:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
                
                endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
                self.r2_client = boto3.client(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name='auto',
                    config=BotoConfig(signature_version='s3v4')
                )
                logger.info(f"Connected to Cloudflare R2 bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cloudflare R2 client: {e}. Falling back to local storage.")
                self.r2_client = None
                self.r2_enabled = False
        else:
            self.r2_enabled = False
            logger.info("Using Local Disk Storage (Cloudflare R2 not enabled or credentials unset).")

    def is_allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def generate_unique_filename(self, original_filename):
        safe_name = secure_filename(original_filename)
        ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'jpg'
        unique_id = uuid.uuid4().hex[:12]
        return f"{unique_id}_{safe_name}"

    def save_file(self, file_storage, custom_filename=None):
        """
        Saves an uploaded file to Cloudflare R2 if configured, or local static/uploads.
        
        Returns a dict:
        {
            "success": True,
            "filename": unique_filename,
            "storage_type": "r2" | "local",
            "url": accessible_url,
            "local_path": local_path_for_inference
        }
        """
        original_filename = file_storage.filename or "upload.jpg"
        if not self.is_allowed_file(original_filename):
            raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
            
        unique_filename = custom_filename or self.generate_unique_filename(original_filename)
        local_dest_path = os.path.join(self.local_upload_folder, unique_filename)
        
        # Save a copy locally so model inference can read it without downloading
        file_storage.seek(0)
        file_storage.save(local_dest_path)
        
        # If Cloudflare R2 is enabled and initialized, upload to R2
        if self.r2_enabled and self.r2_client and self.bucket_name:
            try:
                content_type = file_storage.mimetype or 'image/jpeg'
                r2_key = f"uploads/{unique_filename}"
                
                with open(local_dest_path, 'rb') as f:
                    self.r2_client.put_object(
                        Bucket=self.bucket_name,
                        Key=r2_key,
                        Body=f,
                        ContentType=content_type
                    )
                
                # Construct URL
                if self.public_url:
                    image_url = f"{self.public_url}/{r2_key}"
                else:
                    # Generate a presigned URL valid for 7 days
                    image_url = self.r2_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket_name, 'Key': r2_key},
                        ExpiresIn=604800
                    )
                
                return {
                    "success": True,
                    "filename": unique_filename,
                    "storage_type": "r2",
                    "url": image_url,
                    "local_path": local_dest_path,
                    "r2_key": r2_key
                }
            except Exception as e:
                logger.error(f"Error uploading to Cloudflare R2: {e}. Falling back to local URL.")
        
        # Local fallback return
        local_url = f"/static/uploads/{unique_filename}"
        return {
            "success": True,
            "filename": unique_filename,
            "storage_type": "local",
            "url": local_url,
            "local_path": local_dest_path,
            "r2_key": None
        }

# Global instance for app import
storage_manager = StorageManager()
