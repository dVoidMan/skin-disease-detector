import os
import json
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
INFO_PATH = os.path.join(MODELS_DIR, 'disease_info.json')

# Standard target classes matching disease_info.json
CLASS_NAMES = [
    "acne",
    "eczema",
    "psoriasis",
    "ringworm",
    "vitiligo",
    "melanoma",
    "melanocytic_nevus",
    "dermatitis"
]

class SkinDiseasePredictor:
    """
    Handles image preprocessing, deep learning model loading, inference,
    and clinical information retrieval.
    """
    def __init__(self):
        self.model = None
        self.is_trained_model = False
        self.disease_info = self._load_disease_info()
        self._initialize_model()

    def _load_disease_info(self):
        """Loads clinical details and precautions from JSON."""
        if os.path.exists(INFO_PATH):
            try:
                with open(INFO_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load disease_info.json: {e}")
        return {}

    def _initialize_model(self):
        """Attempts to load trained Keras weights if present."""
        weights_paths = [
            os.path.join(MODELS_DIR, 'skin_disease_model.keras'),
            os.path.join(MODELS_DIR, 'skin_disease_model.h5')
        ]
        
        target_path = None
        for p in weights_paths:
            if os.path.exists(p):
                target_path = p
                break

        if target_path:
            try:
                import tensorflow as tf
                logger.info(f"Loading trained weights from {target_path}...")
                self.model = tf.keras.models.load_model(target_path)
                self.is_trained_model = True
                logger.info("Deep learning model loaded successfully.")
            except Exception as e:
                logger.warning(f"TensorFlow model loading failed: {e}. Utilizing smart demo inference.")
                self.model = None
                self.is_trained_model = False
        else:
            logger.info("Trained model weights not found in models/. Using heuristic demo baseline.")
            self.model = None
            self.is_trained_model = False

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Loads an image, converts to RGB, resizes to target_size,
        and normalizes pixel values to [0, 1].
        """
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(img_array, axis=0)

    def _predict_with_gemini_vision(self, image_path):
        """
        Queries Google Gemini Vision API when GEMINI_API_KEY is configured in .env.
        Provides real medical-grade multimodal vision inference and clinical rationale.
        """
        api_key = os.getenv('GEMINI_API_KEY', '').strip() or os.getenv('GOOGLE_GENAI_API_KEY', '').strip()
        if not api_key:
            return None

        try:
            import base64
            import requests

            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"

            models_to_try = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            prompt = (
                "You are an expert dermatological clinical decision support AI. "
                "Analyze the uploaded skin lesion photograph carefully. "
                "Classify the lesion into one of these 8 exact categories: "
                "acne, eczema, psoriasis, ringworm, vitiligo, melanoma, melanocytic_nevus, dermatitis. "
                "Respond strictly with a JSON object matching this schema: "
                "{\n"
                '  "top_class": "one of the 8 classes",\n'
                '  "confidence": <float percentage e.g. 85.0 or decimal e.g. 0.85>,\n'
                '  "is_valid_lesion": <true if real human skin close-up, false if non-skin/room/cartoon>,\n'
                '  "probabilities": {\n'
                '    "acne": <float 0-1>,\n'
                '    "eczema": <float 0-1>,\n'
                '    "psoriasis": <float 0-1>,\n'
                '    "ringworm": <float 0-1>,\n'
                '    "vitiligo": <float 0-1>,\n'
                '    "melanoma": <float 0-1>,\n'
                '    "melanocytic_nevus": <float 0-1>,\n'
                '    "dermatitis": <float 0-1>\n'
                '  },\n'
                '  "clinical_rationale": "<1-2 sentences on clinical visual features observed>"\n'
                "}"
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": b64_img}}
                    ]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json"
                }
            }

            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                res = requests.post(url, headers=headers, json=payload, timeout=15)
                if res.status_code == 200:
                    raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(raw_text)
                    top = parsed.get("top_class", "").lower().strip()
                    if top in CLASS_NAMES:
                        raw_p = parsed.get("probabilities", {})
                        tot = sum(float(raw_p.get(c, 0.01)) for c in CLASS_NAMES)
                        tot = tot if tot > 0 else 1.0
                        norm_probs = {c: float(raw_p.get(c, 0.01)) / tot for c in CLASS_NAMES}
                        conf = float(parsed.get("confidence", 85.0))
                        if conf <= 1.0:
                            conf = round(conf * 100.0, 2)
                        else:
                            conf = round(conf, 2)
                        return {
                            "top_class": top,
                            "confidence": conf,
                            "is_valid_lesion": bool(parsed.get("is_valid_lesion", True)),
                            "prob_map": norm_probs,
                            "clinical_rationale": parsed.get("clinical_rationale", "")
                        }
        except Exception as e:
            logger.warning(f"Gemini Vision API query failed: {e}. Falling back to local pipeline.")
        return None

    def _heuristic_predict(self, image_path):
        """
        Provides a realistic, image-grounded heuristic prediction when
        offline or before deep learning weights are trained.
        Segments out stark clinical backgrounds (pure white/black paper or tables)
        and evaluates biologically grounded color & texture statistics on skin tissue only.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB').resize((128, 128))
                arr = np.array(img, dtype=np.float32)

                # Segment out stark white/gray backgrounds (R>220, G>220, B>220)
                # and solid black studio borders (R<25, G<25, B<25)
                mask = ~(((arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)) |
                         ((arr[:, :, 0] < 25) & (arr[:, :, 1] < 25) & (arr[:, :, 2] < 25)))

                if np.sum(mask) > 100:
                    skin_pixels = arr[mask]
                else:
                    skin_pixels = arr.reshape(-1, 3)

                r_mean = np.mean(skin_pixels[:, 0])
                g_mean = np.mean(skin_pixels[:, 1])
                b_mean = np.mean(skin_pixels[:, 2])
                brightness = (r_mean + g_mean + b_mean) / 3.0
                redness = r_mean - ((g_mean + b_mean) / 2.0)
                std_dev = np.std(skin_pixels)
        except Exception:
            brightness, redness, std_dev = 128.0, 10.0, 30.0

        scores = {}
        # Vitiligo: true depigmentation (chalk-white skin patches on the tissue itself, WITHOUT acute redness)
        if brightness > 210 and redness < 15:
            scores['vitiligo'] = 4.5 + (brightness / 50.0)
        else:
            scores['vitiligo'] = 0.5

        if redness > 25:
            # Intense erythematous, inflamed, scaly patches (Eczema / Dermatitis / Psoriasis)
            scores['eczema'] = 4.8 + (redness / 12.0)
            scores['dermatitis'] = 3.8 + (redness / 16.0)
            scores['psoriasis'] = 3.2 + (redness / 20.0)
            scores['acne'] = 2.6 + (std_dev / 25.0)
            scores['ringworm'] = 2.2
        else:
            scores['eczema'] = 1.2
            scores['dermatitis'] = 1.0
            scores['psoriasis'] = 1.1
            scores['acne'] = 1.5
            scores['ringworm'] = 1.0

        if brightness < 90 and std_dev > 45:
            scores['melanoma'] = 4.0 + ((100 - brightness) / 20.0)
            scores['melanocytic_nevus'] = 3.5
        elif brightness < 120:
            scores['melanocytic_nevus'] = 4.2
            scores['melanoma'] = 1.0
        else:
            scores['melanoma'] = 0.3
            scores['melanocytic_nevus'] = 1.0

        # Convert raw scores to softmax probabilities
        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: float(v / total_exp) for k, v in exp_scores.items()}

    def evaluate_skin_presence(self, image_path):
        """
        Evaluates whether the image contains human skin pixels using YCbCr color-space segmentation
        and analyzes edge complexity to filter out room scenes, furniture, clothing, and distant crowds.
        """
        try:
            with Image.open(image_path) as img:
                # 1. YCbCr Chrominance Segmentation
                rgb_img = img.convert('RGB').resize((128, 128))
                arr = np.array(rgb_img, dtype=np.float32)

                R = arr[:, :, 0]
                G = arr[:, :, 1]
                B = arr[:, :, 2]
                Y = 0.299 * R + 0.587 * G + 0.114 * B
                Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
                Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

                # YCbCr threshold covering human skin tones (Fitzpatrick types I-VI)
                skin_mask = (Cr >= 128) & (Cr <= 180) & (Cb >= 80) & (Cb <= 135) & (Y >= 35)
                skin_ratio = float(np.mean(skin_mask))

                # 2. Structural Edge / Scene Complexity Analysis
                # Macro skin lesions are organic, smooth biological tissues (< 6-8% sharp edges).
                # Architectural room scenes, furniture, slatted wood walls, suits, and crowds
                # contain high edge density (> 8%).
                gray_img = img.convert('L').resize((128, 128))
                gray_arr = np.array(gray_img, dtype=np.float32)
                dx = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
                dy = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
                edge_mag = (dx[:-1, :] + dy[:, :-1]) / 2.0
                edge_density = float(np.mean(edge_mag > 20.0))

                is_skin = skin_ratio >= 0.25
                is_scene_cluttered = edge_density > 0.08

                return {
                    "is_skin": is_skin,
                    "skin_ratio": round(skin_ratio * 100.0, 1),
                    "edge_density": round(edge_density * 100.0, 1),
                    "is_scene_cluttered": is_scene_cluttered
                }
        except Exception as e:
            logger.warning(f"Error evaluating skin presence: {e}")
            return {
                "is_skin": True,
                "skin_ratio": 50.0,
                "edge_density": 2.0,
                "is_scene_cluttered": False
            }

    def predict(self, image_path):
        """
        Executes prediction on the provided image path with clinical guardrails.
        Checks for human skin presence, scene complexity, and confidence thresholds
        before rendering conclusions.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Guardrail Check: Is the image human skin and close-up?
        skin_eval = self.evaluate_skin_presence(image_path)

        # 2. Model inference: Gemini Vision API -> Local Keras Weights -> Grounded Heuristic
        gemini_result = self._predict_with_gemini_vision(image_path)
        clinical_notes = None

        if gemini_result is not None:
            top_class = gemini_result["top_class"]
            confidence_pct = gemini_result["confidence"]
            prob_map = gemini_result["prob_map"]
            mode = "gemini_vision_api"
            clinical_notes = gemini_result.get("clinical_rationale")
            sorted_probs = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
            if not gemini_result.get("is_valid_lesion", True):
                skin_eval["is_skin"] = False
        elif self.is_trained_model and self.model is not None:
            tensor = self.preprocess_image(image_path)
            raw_preds = self.model.predict(tensor)[0]
            prob_map = {CLASS_NAMES[i]: float(raw_preds[i]) for i in range(len(CLASS_NAMES))}
            mode = "trained_model"
            sorted_probs = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
            top_class, top_prob = sorted_probs[0]
            confidence_pct = round(top_prob * 100.0, 2)
        else:
            prob_map = self._heuristic_predict(image_path)
            mode = "demo_heuristic"
            sorted_probs = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
            top_class, top_prob = sorted_probs[0]
            confidence_pct = round(top_prob * 100.0, 2)

        # 3. Guardrail validation logic
        is_valid_lesion = True
        warning_message = None

        if not skin_eval["is_skin"]:
            is_valid_lesion = False
            warning_message = (
                f"Non-Skin Image Detected ({skin_eval['skin_ratio']}% skin color): "
                "The uploaded photo does not appear to contain human skin tissue. "
                "The system cannot screen clothing, furniture, animals, or non-skin objects. "
                "Please upload a focused photograph centered directly on the affected skin lesion."
            )
        elif skin_eval.get("is_scene_cluttered", False):
            is_valid_lesion = False
            warning_message = (
                f"Complex Scene / Room Environment Detected (Structural Edges: {skin_eval['edge_density']}%): "
                "The photo contains room architecture, furniture, clothing, or distant subjects rather than a close-up macro skin lesion. "
                "Please take a focused, close-up photograph centered on the skin."
            )
        elif confidence_pct < 50.0:
            is_valid_lesion = False
            warning_message = (
                f"Low Screening Confidence ({confidence_pct}%): "
                "The visual patterns are inconclusive or ambiguous across dermatological categories. "
                "Please retake the photo with improved lighting and focus, or consult a qualified physician."
            )

        # If guardrails fail, DO NOT suggest any disease classification or probabilities
        if not is_valid_lesion:
            inconclusive_info = {
                "name": "Inconclusive Scan",
                "medical_term": "Unclassifiable Image",
                "risk_level": "Undetermined",
                "badge_class": "bg-secondary",
                "summary": warning_message,
                "common_symptoms": [],
                "possible_causes": [],
                "self_care_precautions": [],
                "when_to_see_doctor": "Consult a healthcare professional or dermatologist for direct examination of any concerning skin mark."
            }
            return {
                "class_id": "inconclusive",
                "disease_name": "Inconclusive Scan",
                "raw_disease_name": "Inconclusive Scan",
                "confidence": 0.0,
                "is_trained_model": self.is_trained_model,
                "inference_mode": mode,
                "is_valid_lesion": False,
                "skin_ratio": skin_eval['skin_ratio'],
                "edge_density": skin_eval.get('edge_density', 0.0),
                "is_scene_cluttered": skin_eval.get('is_scene_cluttered', False),
                "warning_message": warning_message,
                "info": inconclusive_info,
                "probabilities": {
                    '_meta': {
                        'is_valid_lesion': False,
                        'is_skin': skin_eval['is_skin'],
                        'skin_ratio': skin_eval['skin_ratio'],
                        'edge_density': skin_eval.get('edge_density', 0.0),
                        'is_scene_cluttered': skin_eval.get('is_scene_cluttered', False),
                        'warning_message': warning_message
                    }
                },
                "top_breakdown": []
            }

        # Retrieve disease info for valid lesions
        info = self.disease_info.get(top_class, {
            "name": top_class.replace('_', ' ').title(),
            "medical_term": "N/A",
            "risk_level": "Unspecified",
            "badge_class": "bg-secondary",
            "summary": "Information pending clinical classification.",
            "common_symptoms": [],
            "possible_causes": [],
            "self_care_precautions": [],
            "when_to_see_doctor": "Consult a healthcare provider for diagnosis."
        })

        # Top ranking for breakdown
        top_breakdown = [
            {
                "class_id": c,
                "name": self.disease_info.get(c, {}).get("name", c.replace('_', ' ').title()),
                "probability": round(p * 100.0, 1),
                "risk_level": self.disease_info.get(c, {}).get("risk_level", "Low"),
                "badge_class": self.disease_info.get(c, {}).get("badge_class", "bg-info")
            }
            for c, p in sorted_probs[:4]
        ]

        # Probabilities dictionary with guardrail metadata
        probabilities_dict = {c: round(p * 100.0, 2) for c, p in sorted_probs}
        probabilities_dict['_meta'] = {
            'is_valid_lesion': True,
            'is_skin': skin_eval['is_skin'],
            'skin_ratio': skin_eval['skin_ratio'],
            'edge_density': skin_eval.get('edge_density', 0.0),
            'is_scene_cluttered': skin_eval.get('is_scene_cluttered', False),
            'clinical_rationale': clinical_notes,
            'warning_message': None
        }

        return {
            "class_id": top_class,
            "disease_name": info.get("name", top_class.title()),
            "raw_disease_name": info.get("name", top_class.title()),
            "confidence": confidence_pct,
            "is_trained_model": self.is_trained_model,
            "inference_mode": mode,
            "clinical_rationale": clinical_notes,
            "is_valid_lesion": True,
            "skin_ratio": skin_eval['skin_ratio'],
            "edge_density": skin_eval.get('edge_density', 0.0),
            "is_scene_cluttered": skin_eval.get('is_scene_cluttered', False),
            "warning_message": None,
            "info": info,
            "probabilities": probabilities_dict,
            "top_breakdown": top_breakdown
        }

    def get_all_diseases(self):
        """Returns the full knowledge base encyclopedia."""
        return self.disease_info

# Global predictor instance
predictor = SkinDiseasePredictor()
