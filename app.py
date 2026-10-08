import os
import json
import logging
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, abort
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from config import config_by_name
from database import db, bcrypt, login_manager
from database.models import User, PredictionHistory
from storage import storage_manager
from models.predictor import predictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """Application factory function."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, config_by_name['default']))

    # Initialize extensions
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    storage_manager.init_app(app)

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Create tables and directories within app context
    with app.app_context():
        # Ensure database directory exists before db.create_all()
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
        if db_uri.startswith('sqlite:'):
            db_path = getattr(db.engine.url, 'database', None)
            if db_path:
                db_dir = os.path.dirname(os.path.abspath(db_path))
                os.makedirs(db_dir, exist_ok=True)

        os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)
        os.makedirs(app.config.get('UPLOAD_FOLDER'), exist_ok=True)
        db.create_all()
        logger.info("Database initialized successfully.")

    # -------------------------------------------------------------
    # Routes: Main Pages
    # -------------------------------------------------------------
    @app.route('/')
    def index():
        diseases = predictor.get_all_diseases()
        return render_template('index.html', diseases=diseases)

    @app.route('/detect')
    def detect():
        return render_template('index.html', scroll_to_upload=True)

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            flash('No image file selected. Please choose a skin image to analyze.', 'danger')
            return redirect(url_for('index'))

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(url_for('index'))

        try:
            # 1. Save file via unified storage (Cloudflare R2 or Local)
            upload_result = storage_manager.save_file(file)
            
            # 2. Run deep learning prediction / inference
            pred_result = predictor.predict(upload_result['local_path'])

            # 3. Store prediction in database
            user_id = current_user.id if current_user.is_authenticated else None
            patient_notes = request.form.get('notes', '').strip()

            record = PredictionHistory(
                user_id=user_id,
                image_url=upload_result['url'],
                storage_type=upload_result['storage_type'],
                r2_key=upload_result.get('r2_key'),
                disease_name=pred_result['disease_name'],
                confidence=pred_result['confidence'],
                probabilities_json=json.dumps(pred_result['probabilities']),
                patient_notes=patient_notes
            )
            db.session.add(record)
            db.session.commit()

            # If client made an AJAX request, respond with JSON
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
                return jsonify({
                    'success': True,
                    'redirect_url': url_for('result', id=record.id),
                    'prediction': pred_result
                })

            if not pred_result.get('is_valid_lesion', True):
                flash(pred_result.get('warning_message', 'Image scan is inconclusive.'), 'warning')
            else:
                flash('Skin lesion analyzed successfully.', 'success')
            return redirect(url_for('result', id=record.id))

        except ValueError as ve:
            flash(str(ve), 'warning')
            return redirect(url_for('index'))
        except Exception as e:
            logger.exception("Error during prediction:")
            flash(f"An error occurred while analyzing the image: {str(e)}", 'danger')
            return redirect(url_for('index'))

    @app.route('/result/<int:id>')
    def result(id):
        record = PredictionHistory.query.get_or_404(id)
        
        # Filter out metadata from probabilities
        raw_probs = record.probabilities or {}
        meta = raw_probs.get('_meta', {})
        is_valid = meta.get('is_valid_lesion', True) and not record.disease_name.startswith('Inconclusive') and record.confidence >= 50.0

        if not is_valid:
            info = {
                "name": "Inconclusive Scan",
                "medical_term": "Unclassifiable Image",
                "risk_level": "Undetermined",
                "badge_class": "bg-secondary",
                "summary": meta.get('warning_message') or "The uploaded image does not meet clinical standards for skin lesion screening. No classification was performed.",
                "common_symptoms": [],
                "possible_causes": [],
                "self_care_precautions": [],
                "when_to_see_doctor": "Consult a healthcare provider or dermatologist for direct examination of any concerning skin mark."
            }
            top_breakdown = []
        else:
            # Look up disease info for valid lesions
            all_info = predictor.get_all_diseases()
            lookup_name = record.disease_name.strip()
            disease_key = lookup_name.lower().replace(' ', '_')
            info = all_info.get(disease_key)
            
            # If not found directly, find by display name
            if not info:
                for k, v in all_info.items():
                    if v.get('name', '').lower() == lookup_name.lower():
                        info = v
                        break

            numeric_probs = {k: v for k, v in raw_probs.items() if not k.startswith('_') and isinstance(v, (int, float))}
            top_breakdown = [
                {
                    'name': all_info.get(c, {}).get('name', c.replace('_', ' ').title()),
                    'probability': p,
                    'risk_level': all_info.get(c, {}).get('risk_level', 'Low'),
                    'badge_class': all_info.get(c, {}).get('badge_class', 'bg-info')
                }
                for c, p in sorted(numeric_probs.items(), key=lambda x: x[1], reverse=True)[:4]
            ]

        return render_template(
            'result.html',
            record=record,
            info=info or {},
            meta=meta,
            is_valid=is_valid,
            top_breakdown=top_breakdown,
            is_trained_model=predictor.is_trained_model
        )

    @app.route('/history')
    @login_required
    def history():
        user_scans = PredictionHistory.query.filter_by(user_id=current_user.id)\
            .order_by(PredictionHistory.created_at.desc()).all()
        return render_template('history.html', scans=user_scans)

    @app.route('/info')
    def encyclopedia():
        diseases = predictor.get_all_diseases()
        return render_template('info.html', diseases=diseases)

    @app.route('/about')
    def about():
        return render_template('about.html')

    # -------------------------------------------------------------
    # Routes: Authentication
    # -------------------------------------------------------------
    @app.route('/auth/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('index'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')

            if not username or not email or not password:
                flash('Please fill in all required fields.', 'warning')
                return render_template('auth/register.html')

            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('auth/register.html')

            if User.query.filter_by(username=username).first():
                flash('Username is already registered. Please choose another.', 'warning')
                return render_template('auth/register.html')

            if User.query.filter_by(email=email).first():
                flash('An account with this email already exists.', 'warning')
                return render_template('auth/register.html')

            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            login_user(new_user)
            flash('Account created successfully! Welcome to Skin Health AI.', 'success')
            return redirect(url_for('index'))

        return render_template('auth/register.html')

    @app.route('/auth/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))

        if request.method == 'POST':
            username_or_email = request.form.get('username_or_email', '').strip()
            password = request.form.get('password', '')
            remember = True if request.form.get('remember') else False

            user = User.query.filter(
                (User.username == username_or_email) | (User.email == username_or_email.lower())
            ).first()

            if user and user.check_password(password):
                login_user(user, remember=remember)
                flash(f'Welcome back, {user.username}!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                flash('Invalid username/email or password.', 'danger')

        return render_template('auth/login.html')

    @app.route('/auth/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out safely.', 'info')
        return redirect(url_for('index'))

    # -------------------------------------------------------------
    # Routes: REST API Endpoints
    # -------------------------------------------------------------
    @app.route('/api/v1/predict', methods=['POST'])
    def api_predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        file = request.files['image']
        try:
            upload_result = storage_manager.save_file(file)
            pred_result = predictor.predict(upload_result['local_path'])

            record = PredictionHistory(
                user_id=current_user.id if current_user.is_authenticated else None,
                image_url=upload_result['url'],
                storage_type=upload_result['storage_type'],
                r2_key=upload_result.get('r2_key'),
                disease_name=pred_result['disease_name'],
                confidence=pred_result['confidence'],
                probabilities_json=json.dumps(pred_result['probabilities'])
            )
            db.session.add(record)
            db.session.commit()

            return jsonify({
                'id': record.id,
                'image_url': upload_result['url'],
                'storage_type': upload_result['storage_type'],
                'prediction': pred_result
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/diseases', methods=['GET'])
    def api_diseases():
        return jsonify(predictor.get_all_diseases())

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'service': 'skin-disease-detection'}), 200

    return app

# Application entrypoint instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
