from datetime import datetime
import json
from flask_login import UserMixin
from database import db, bcrypt

class User(db.Model, UserMixin):
    """User account model for authentication and personal scan history."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    predictions = db.relationship('PredictionHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class PredictionHistory(db.Model):
    """Stores scan history, image links (local or R2), and AI classification scores."""
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    image_url = db.Column(db.String(500), nullable=False)
    storage_type = db.Column(db.String(20), default='local')  # 'r2' or 'local'
    r2_key = db.Column(db.String(255), nullable=True)
    
    disease_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)  # Stored as percentage e.g. 95.4
    probabilities_json = db.Column(db.Text, nullable=True)  # JSON representation of all classes
    patient_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    @property
    def probabilities(self):
        if self.probabilities_json:
            try:
                return json.loads(self.probabilities_json)
            except Exception:
                return {}
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_url': self.image_url,
            'storage_type': self.storage_type,
            'disease_name': self.disease_name,
            'confidence': round(self.confidence, 2),
            'probabilities': self.probabilities,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

    def __repr__(self):
        return f"<PredictionHistory #{self.id}: {self.disease_name} ({self.confidence}%)>"
