import os
import shutil
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

# --- 1. SELF-HEALING DIRECTORY SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Force create the templates folder if it doesn't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Hunt down the triage.html file wherever it might be hiding
target_html = os.path.join(TEMPLATE_DIR, 'triage.html')
possible_locations = [
    os.path.join(BASE_DIR, 'triage.html'),         # Is it next to app.py?
    os.path.join(PROJECT_ROOT, 'triage.html'),     # Is it in the root folder?
]

if not os.path.exists(target_html):
    for loc in possible_locations:
        if os.path.exists(loc):
            shutil.copy(loc, target_html)
            print(f"🔧 AUTO-FIX: Successfully found triage.html and moved it to {TEMPLATE_DIR}!")
            break

# --- 2. FLASK INITIALIZATION ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# --- 3. LOAD AI MODEL ---
model_path = os.path.join(PROJECT_ROOT, 'models', 'covid_risk_model.pkl')
try:
    model = joblib.load(model_path)
    print("✅ API Server: Medical Triage Model Loaded Successfully")
except FileNotFoundError:
    print(f"❌ ERROR: Model not found. Looked in: {model_path}")
    model = None

AGE_MEAN = 41.79
AGE_STD = 16.90

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h1>Still missing!</h1><p>Flask error: {str(e)}</p><p>Please ensure triage.html is saved somewhere in your project folder.</p>"

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/triage')
def triage():
    return render_template('triage.html')

@app.route('/api/predict_risk', methods=['POST', 'OPTIONS'])
def predict_risk():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        data = request.get_json()

        usmer = float(data.get('USMER', 0))
        medical_unit = float(data.get('MEDICAL_UNIT', 1))
        sex = float(data.get('SEX', 0))
        patient_type = float(data.get('PATIENT_TYPE', 1))
        intubed = float(data.get('INTUBED', 0))
        pneumonia = float(data.get('PNEUMONIA', 0))
        
        raw_age = float(data.get('AGE', 40))
        scaled_age = (raw_age - AGE_MEAN) / AGE_STD
        
        pregnant = float(data.get('PREGNANT', 0))
        diabetes = float(data.get('DIABETES', 0))
        copd = float(data.get('COPD', 0))
        asthma = float(data.get('ASTHMA', 0))
        inmsupr = float(data.get('INMSUPR', 0))
        hipertension = float(data.get('HIPERTENSION', 0))
        other_disease = float(data.get('OTHER_DISEASE', 0))
        cardiovascular = float(data.get('CARDIOVASCULAR', 0))
        obesity = float(data.get('OBESITY', 0))
        renal_chronic = float(data.get('RENAL_CHRONIC', 0))
        tobacco = float(data.get('TOBACCO', 0))
        clasificacion_final = float(data.get('CLASIFICACION_FINAL', 3))
        icu = float(data.get('ICU', 0))

        features = np.array([[
            usmer, medical_unit, sex, patient_type, intubed, pneumonia, 
            scaled_age, pregnant, diabetes, copd, asthma, inmsupr, 
            hipertension, other_disease, cardiovascular, obesity, 
            renal_chronic, tobacco, clasificacion_final, icu
        ]])

        probabilities = model.predict_proba(features)[0]
        mortality_risk = probabilities[1] 
        
        is_high_risk = bool(mortality_risk >= 0.50)

        return jsonify({
            'success': True,
            'risk_level': 'CRITICAL' if is_high_risk else 'STABLE',
            'mortality_probability': round(mortality_risk, 4),
            'risk_score_100': int(mortality_risk * 100)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Medical Triage API on http://localhost:5000")
    app.run(debug=True, port=5000)