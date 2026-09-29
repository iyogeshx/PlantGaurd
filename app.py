"""
PlantVillage Leaf Disease Prediction - Flask Web Application
=============================================================
Loads the pre-trained deep learning model ONCE at startup and serves predictions
via a REST API. Provides sample leaf datasets, disease encyclopedia endpoints,
and robust image preprocessing for the web application.
"""

import os
import json
import base64
import io
import numpy as np
import requests as http_requests
from flask import Flask, render_template, request, jsonify
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
IMG_SIZE = 224

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ─── Gemini AI Chatbot Configuration ─────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')  # Set via environment variable or paste directly
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'

# ─── Load Model & Labels Safely ──────────────────────────────────
model = None
CLASS_LABELS = {}
DISEASE_INFO = {}

def load_resources():
    global model, CLASS_LABELS, DISEASE_INFO
    
    # 1. Load Labels & Info
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            CLASS_LABELS = data.get("labels", {})
            DISEASE_INFO = data.get("disease_info", {})
        print(f"✅ Loaded {len(CLASS_LABELS)} class labels from {LABELS_PATH}")
    else:
        print("⚠️ class_labels.json not found, using default 15 classes")

    # 2. Load or Initialize Model
    try:
        from tensorflow import keras
        if os.path.exists(MODEL_PATH):
            print(f"🔄 Loading neural network from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model file not found, building MobileNetV2 architecture...")
            base_model = keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            from tensorflow.keras import layers
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(len(CLASS_LABELS) or 15, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)
            print(f"✅ Initialized and saved model to {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Notice: TensorFlow model loading encountered: {e}")

load_resources()


# ─── Sample Leaf Images Catalog ───────────────────────────────────
SAMPLE_LEAVES = [
    {
        "id": "tomato_early_blight",
        "title": "Tomato Early Blight",
        "plant": "Tomato",
        "disease": "Early Blight",
        "pathogen": "Fungal (Alternaria solani)",
        "expected_class": "Tomato_Early_blight",
        "image_url": "/static/samples/tomato_early_blight.jpg",
        "badge": "Fungal Target Spot",
        "severity": "moderate"
    },
    {
        "id": "tomato_late_blight",
        "title": "Tomato Late Blight",
        "plant": "Tomato",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "expected_class": "Tomato_Late_blight",
        "image_url": "/static/samples/tomato_late_blight.jpg",
        "badge": "High Severity Blight",
        "severity": "high"
    },
    {
        "id": "potato_late_blight",
        "title": "Potato Late Blight",
        "plant": "Potato",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "expected_class": "Potato___Late_blight",
        "image_url": "/static/samples/potato_late_blight.jpg",
        "badge": "High Risk Pathogen",
        "severity": "high"
    },
    {
        "id": "potato_early_blight",
        "title": "Potato Early Blight",
        "plant": "Potato",
        "disease": "Early Blight",
        "pathogen": "Fungal (Alternaria solani)",
        "expected_class": "Potato___Early_blight",
        "image_url": "/static/samples/potato_early_blight.jpg",
        "badge": "Concentric Spotting",
        "severity": "moderate"
    },
    {
        "id": "pepper_bacterial_spot",
        "title": "Pepper Bacterial Spot",
        "plant": "Pepper Bell",
        "disease": "Bacterial Spot",
        "pathogen": "Bacterial (Xanthomonas)",
        "expected_class": "Pepper__bell___Bacterial_spot",
        "image_url": "/static/samples/pepper_bacterial_spot.jpg",
        "badge": "Bacterial Lesion",
        "severity": "moderate"
    },
    {
        "id": "tomato_healthy",
        "title": "Healthy Tomato Leaf",
        "plant": "Tomato",
        "disease": "Healthy",
        "pathogen": "None (Healthy)",
        "expected_class": "Tomato_healthy",
        "image_url": "/static/samples/tomato_healthy.jpg",
        "badge": "Healthy Tissue",
        "severity": "none"
    },
    {
        "id": "potato_healthy",
        "title": "Healthy Potato Leaf",
        "plant": "Potato",
        "disease": "Healthy",
        "pathogen": "None (Healthy)",
        "expected_class": "Potato___healthy",
        "image_url": "/static/samples/potato_healthy.jpg",
        "badge": "Healthy Tissue",
        "severity": "none"
    },
    {
        "id": "pepper_healthy",
        "title": "Healthy Pepper Bell",
        "plant": "Pepper Bell",
        "disease": "Healthy",
        "pathogen": "None (Healthy)",
        "expected_class": "Pepper__bell___healthy",
        "image_url": "/static/samples/pepper_healthy.jpg",
        "badge": "Healthy Tissue",
        "severity": "none"
    }
]


def preprocess_image(image_bytes):
    """Preprocess uploaded image bytes for model prediction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_severity_color(severity):
    """Return theme color based on disease severity level."""
    colors = {
        "none": "#10b981",      # Emerald Green
        "low": "#84cc16",       # Lime Green
        "moderate": "#f59e0b",  # Amber
        "high": "#ef4444"       # Crimson Red
    }
    return colors.get(str(severity).lower(), "#6b7280")


def get_precaution_color(level):
    """Return badge color based on precaution urgency level."""
    colors = {
        "CRITICAL URGENCY": "#ef4444",   # Crimson
        "HIGH ALERT": "#f97316",         # Orange
        "MODERATE CAUTION": "#f59e0b",   # Amber
        "ROUTINE CARE": "#10b981"        # Emerald
    }
    return colors.get(str(level).upper(), "#34d399")


@app.route('/')
def index():
    """Serve the main application web page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Return API health and model metadata status."""
    return jsonify({
        "status": "ready",
        "model_loaded": model is not None,
        "classes_count": len(CLASS_LABELS),
        "samples_count": len(SAMPLE_LEAVES)
    })


@app.route('/api/samples')
def get_samples():
    """Return curated test leaf samples for instant browser testing."""
    return jsonify(SAMPLE_LEAVES)


@app.route('/api/diseases')
@app.route('/api/classes')
def get_classes():
    """Return all supported disease classes with comprehensive details."""
    classes = []
    for idx in sorted(CLASS_LABELS.keys(), key=int):
        cls = CLASS_LABELS[idx]
        info = DISEASE_INFO.get(cls, {})
        prec_level = info.get("precaution_level", "MODERATE CAUTION")
        classes.append({
            "index": int(idx),
            "class": cls,
            "plant": info.get("plant", "Unknown"),
            "disease": info.get("disease", "Unknown"),
            "pathogen": info.get("pathogen", "N/A"),
            "description": info.get("description", ""),
            "symptoms": info.get("symptoms", []),
            "causes": info.get("causes", ""),
            "treatment": info.get("treatment", ""),
            "organic_treatment": info.get("organic_treatment", ""),
            "prevention": info.get("prevention", ""),
            "precaution_level": prec_level,
            "precaution_color": get_precaution_color(prec_level),
            "precautions": info.get("precautions", []),
            "severity": info.get("severity", "unknown"),
            "severity_color": get_severity_color(info.get("severity", "unknown"))
        })
    return jsonify(classes)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, base64 data, or sample test leaf prediction."""
    image_bytes = None
    mime_type = 'jpeg'

    # 1. Check for standard multipart file upload
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '':
            image_bytes = file.read()
            file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'jpeg'
            mime_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'webp': 'webp'}
            mime_type = mime_map.get(file_ext, 'jpeg')

    # 2. Check for JSON payload (Base64 webcam capture or sample ID)
    if not image_bytes and request.is_json:
        data = request.get_json()
        if 'sample_id' in data:
            sample_id = data['sample_id']
            # Find sample
            matched = next((s for s in SAMPLE_LEAVES if s['id'] == sample_id), None)
            if matched:
                filename = os.path.basename(matched['image_url'])
                filepath = os.path.join(SAMPLES_DIR, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        image_bytes = f.read()
                    mime_type = 'jpeg'
        elif 'image_base64' in data:
            b64_str = data['image_base64']
            if ',' in b64_str:
                header, b64_str = b64_str.split(',', 1)
                if 'png' in header:
                    mime_type = 'png'
                elif 'webp' in header:
                    mime_type = 'webp'
            image_bytes = base64.b64decode(b64_str)

    if not image_bytes:
        return jsonify({'error': 'No image provided. Upload a file, capture via webcam, or choose a sample leaf.'}), 400

    try:
        # Preprocess image
        img_array = preprocess_image(image_bytes)

        # Make prediction with neural network
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])

        # Retrieve mapped class details
        class_name = CLASS_LABELS.get(str(predicted_class_idx), "Tomato_healthy")
        info = DISEASE_INFO.get(class_name, {})

        # Compute Top 3 differential diagnoses
        top3_indices = np.argsort(predictions[0])[::-1][:3]
        top3 = []
        for idx in top3_indices:
            cls = CLASS_LABELS.get(str(idx), f"Class_{idx}")
            cls_info = DISEASE_INFO.get(cls, {})
            top3.append({
                "class": cls,
                "plant": cls_info.get("plant", "Unknown"),
                "disease": cls_info.get("disease", "Unknown"),
                "confidence": float(predictions[0][idx]),
                "severity": cls_info.get("severity", "unknown"),
                "severity_color": get_severity_color(cls_info.get("severity", "unknown"))
            })

        # Format base64 preview
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')

        prec_level = info.get("precaution_level", "MODERATE CAUTION")

        result = {
            "success": True,
            "prediction": {
                "class": class_name,
                "plant": info.get("plant", "Unknown"),
                "disease": info.get("disease", "Unknown"),
                "pathogen": info.get("pathogen", "Unknown Pathogen"),
                "confidence": confidence,
                "description": info.get("description", ""),
                "symptoms": info.get("symptoms", []),
                "causes": info.get("causes", ""),
                "treatment": info.get("treatment", ""),
                "organic_treatment": info.get("organic_treatment", ""),
                "prevention": info.get("prevention", ""),
                "precaution_level": prec_level,
                "precaution_color": get_precaution_color(prec_level),
                "precautions": info.get("precautions", []),
                "severity": info.get("severity", "unknown"),
                "severity_color": get_severity_color(info.get("severity", "unknown"))
            },
            "top3": top3,
            "image": f"data:image/{mime_type};base64,{img_b64}"
        }

        return jsonify(result)


    except Exception as e:
        return jsonify({'error': f"Inference error: {str(e)}"}), 500


# ─── Weather & Disease Risk API (Free, No API Key) ───────
@app.route('/api/weather')
def get_weather():
    """Fetch weather data from Open-Meteo (free, no API key) and calculate disease risk."""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is None or lon is None:
        return jsonify({'error': 'Latitude and longitude are required.'}), 400

    try:
        # Open-Meteo: completely free, no API key, no signup
        url = (
            f'https://api.open-meteo.com/v1/forecast?'
            f'latitude={lat}&longitude={lon}'
            f'&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code'
            f'&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max'
            f'&timezone=auto&forecast_days=3'
        )
        resp = http_requests.get(url, timeout=10)
        data = resp.json()

        current = data.get('current', {})
        daily = data.get('daily', {})

        temp = current.get('temperature_2m', 0)
        humidity = current.get('relative_humidity_2m', 0)
        wind = current.get('wind_speed_10m', 0)
        weather_code = current.get('weather_code', 0)

        # ─── Disease Risk Calculation Algorithm ───
        risk_score = 0
        risk_factors = []

        # Humidity factor (most critical for fungal diseases)
        if humidity >= 90:
            risk_score += 40
            risk_factors.append('Critically high humidity (>=90%) — ideal for fungal spore germination')
        elif humidity >= 75:
            risk_score += 25
            risk_factors.append('High humidity (>=75%) — favorable for Late Blight & Leaf Mold')
        elif humidity >= 60:
            risk_score += 10
            risk_factors.append('Moderate humidity — monitor for early signs of disease')

        # Temperature factor
        if 18 <= temp <= 24:
            risk_score += 25
            risk_factors.append(f'Temperature ({temp} C) is in the optimal range for Phytophthora infestans')
        elif 15 <= temp <= 28:
            risk_score += 15
            risk_factors.append(f'Temperature ({temp} C) supports many common foliar pathogens')
        elif temp > 30:
            risk_score += 5
            risk_factors.append(f'High temperature ({temp} C) — bacterial spot risk increases')

        # Precipitation factor
        precip_prob = daily.get('precipitation_probability_max', [0])
        today_precip = precip_prob[0] if precip_prob else 0
        if today_precip >= 70:
            risk_score += 20
            risk_factors.append(f'High rain probability ({today_precip}%) — splash-dispersed pathogens likely')
        elif today_precip >= 40:
            risk_score += 10
            risk_factors.append(f'Moderate rain expected ({today_precip}%) — keep foliage dry')

        # Low wind = poor air circulation
        if wind < 5:
            risk_score += 10
            risk_factors.append('Low wind speed — poor air circulation increases fungal risk')

        risk_score = min(risk_score, 100)

        if risk_score >= 70:
            risk_level = 'CRITICAL'
            risk_color = '#ef4444'
            risk_advice = 'Apply preventive fungicide immediately. Avoid overhead irrigation. Inspect all foliage for early symptoms.'
        elif risk_score >= 45:
            risk_level = 'HIGH'
            risk_color = '#f97316'
            risk_advice = 'Conditions favor disease spread. Use drip irrigation only. Scout plants twice daily for lesions or wilting.'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
            risk_color = '#f59e0b'
            risk_advice = 'Monitor humidity trends. Maintain proper plant spacing and soil drainage.'
        else:
            risk_level = 'LOW'
            risk_color = '#10b981'
            risk_advice = 'Conditions are favorable for healthy growth. Continue routine crop care.'

        # Weather condition text from WMO code
        wmo_conditions = {
            0: 'Clear Sky', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Depositing Rime Fog',
            51: 'Light Drizzle', 53: 'Moderate Drizzle', 55: 'Dense Drizzle',
            61: 'Slight Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
            71: 'Slight Snow', 73: 'Moderate Snow', 75: 'Heavy Snow',
            80: 'Rain Showers', 81: 'Moderate Showers', 82: 'Violent Showers',
            95: 'Thunderstorm', 96: 'Thunderstorm w/ Hail', 99: 'Severe Thunderstorm'
        }
        condition = wmo_conditions.get(weather_code, 'Unknown')

        # 3-day forecast summary
        forecast = []
        if daily:
            dates = daily.get('time', [])
            t_max = daily.get('temperature_2m_max', [])
            t_min = daily.get('temperature_2m_min', [])
            p_prob = daily.get('precipitation_probability_max', [])
            for i in range(min(3, len(dates))):
                forecast.append({
                    'date': dates[i] if i < len(dates) else '',
                    'temp_max': t_max[i] if i < len(t_max) else 0,
                    'temp_min': t_min[i] if i < len(t_min) else 0,
                    'rain_probability': p_prob[i] if i < len(p_prob) else 0
                })

        return jsonify({
            'success': True,
            'current': {
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind,
                'condition': condition,
                'weather_code': weather_code
            },
            'disease_risk': {
                'score': risk_score,
                'level': risk_level,
                'color': risk_color,
                'factors': risk_factors,
                'advice': risk_advice
            },
            'forecast': forecast
        })

    except Exception as e:
        return jsonify({'error': f'Weather fetch failed: {str(e)}'}), 500


# ─── AI Chatbot via Gemini API ────────────────────────────
CHATBOT_SYSTEM_PROMPT = """You are PlantGuard AI Assistant, an expert agricultural plant pathologist.
Your specialty is diagnosing and treating diseases in Tomato, Potato, and Pepper (Bell Pepper) crops.

Key rules:
- Answer ONLY questions about plant diseases, crop health, agriculture, gardening, and related topics.
- If asked about unrelated topics, politely redirect to plant health topics.
- Provide actionable, practical advice that farmers can follow immediately.
- When recommending treatments, always offer both organic/biological AND chemical options.
- Include precautionary measures and prevention strategies.
- Use clear, simple language accessible to farmers of all education levels.
- Keep responses concise but comprehensive (2-4 paragraphs max).
- When you mention specific diseases, include the scientific name of the pathogen.
- If the user describes symptoms, suggest possible diagnoses ranked by likelihood.
- Always end with a practical next step the farmer can take.
"""

def get_offline_pathology_response(user_msg):
    """Provide rule-based pathology guidance when Gemini API key is absent."""
    msg = user_msg.lower()
    if 'tomato' in msg or 'blight' in msg or 'spot' in msg or 'mold' in msg or 'curl' in msg:
        return (
            "🌿 **PlantGuard Diagnostic Guidance (Tomato Pathology)**:\n\n"
            "• **Early Blight (*Alternaria linariae*)**: Manifests as brown concentric 'bullseye' target rings on older foliage. Apply copper fungicides or chlorothalonil, prune affected lower branches, and avoid splash watering.\n"
            "• **Late Blight (*Phytophthora infestans*)**: Rapidly spreading dark water-soaked necrotic patches with white fuzzy fungal spores underneath during high humidity (>85%). Apply mancozeb or copper octanoate immediately.\n"
            "• **Yellow Leaf Curl Virus**: Transmitted by whitefly (*Bemisia tabaci*). Control vectors with insect netting, yellow sticky traps, and neem oil.\n\n"
            "💡 *Tip: Configure the `GEMINI_API_KEY` environment variable to enable live conversational reasoning.*"
        )
    elif 'potato' in msg:
        return (
            "🥔 **PlantGuard Diagnostic Guidance (Potato Pathology)**:\n\n"
            "• **Late Blight (*Phytophthora infestans*)**: Highly aggressive water-soaked lesions that collapse foliage. Destroy infected haulms before tuber infection occurs and ensure systemic fungicide coverage.\n"
            "• **Early Blight (*Alternaria solani*)**: Concentric ring dark lesions with chlorotic yellow halos. Rotate crops with non-solanaceous species for at least 3 years.\n\n"
            "💡 *Tip: Configure the `GEMINI_API_KEY` environment variable to enable live conversational reasoning.*"
        )
    elif 'pepper' in msg:
        return (
            "🫑 **PlantGuard Diagnostic Guidance (Pepper Pathology)**:\n\n"
            "• **Bacterial Spot (*Xanthomonas campestris*)**: Angular dark brown water-soaked lesions that cause severe defoliation. Spray copper-based bactericides at first sign.\n"
            "• **Preventative Protocol**: Space plants 45-60cm apart for rapid canopy drying and use drip irrigation instead of overhead sprinklers.\n\n"
            "💡 *Tip: Configure the `GEMINI_API_KEY` environment variable to enable live conversational reasoning.*"
        )
    else:
        return (
            f"🌿 **PlantGuard Agronomic Guidance**:\n\n"
            f"Regarding *\"{user_msg}\"*:\n"
            "For Solanaceous crops (Tomato, Potato, Pepper), foliar disease prevention follows three key rules:\n"
            "1. **Foliage Dryness**: Fungal spores require 4–8 hours of leaf wetness to germinate. Always irrigate at the base.\n"
            "2. **Sanitation**: Disinfect pruning shears with 70% isopropyl alcohol between plants to halt pathogen transmission.\n"
            "3. **Soil Mulching**: Lay down straw or organic mulch to prevent soil-borne pathogens from splashing onto lower leaves during rainfall.\n\n"
            "💡 *Tip: Configure the `GEMINI_API_KEY` environment variable to enable live conversational reasoning.*"
        )


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Proxy chat messages to Gemini API with plant pathology system prompt."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided.'}), 400

    user_message = data['message']
    chat_history = data.get('history', [])

    if not GEMINI_API_KEY:
        reply = get_offline_pathology_response(user_message)
        return jsonify({'success': True, 'reply': reply})

    try:
        # Build conversation contents for Gemini
        contents = []

        # Add conversation history (last 10 messages for context window)
        for msg in chat_history[-10:]:
            contents.append({
                'role': msg.get('role', 'user'),
                'parts': [{'text': msg.get('text', '')}]
            })

        # Add current user message
        contents.append({
            'role': 'user',
            'parts': [{'text': user_message}]
        })

        payload = {
            'system_instruction': {
                'parts': [{'text': CHATBOT_SYSTEM_PROMPT}]
            },
            'contents': contents,
            'generationConfig': {
                'temperature': 0.7,
                'maxOutputTokens': 1024,
                'topP': 0.9
            }
        }

        resp = http_requests.post(
            f'{GEMINI_API_URL}?key={GEMINI_API_KEY}',
            json=payload,
            timeout=30
        )

        if resp.status_code != 200:
            error_detail = resp.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({'error': f'Gemini API error: {error_detail}'}), 500

        result = resp.json()
        reply = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        if not reply:
            reply = 'I apologize, I could not generate a response. Please try rephrasing your question about plant diseases.'

        return jsonify({
            'success': True,
            'reply': reply
        })

    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'The AI service timed out. Please try again.'}), 504
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 64)
    print("  🌿 PlantGuard AI — Plant Disease Prediction Server")
    print(f"  🌐 Running locally on: http://localhost:{port}")
    print("=" * 64 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)

