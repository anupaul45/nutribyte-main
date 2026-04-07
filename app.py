from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file,flash
from flask_cors import CORS
import os, json, joblib, numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime
import tempfile
from diet_planner import DietPlanner
import re
import pandas as pd
import pytesseract
from werkzeug.utils import secure_filename
from PIL import Image


# -------------------- Initialization --------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
CORS(app)

# -------------------- File paths --------------------
USERS_FILE = "users.json"
USER_DATA_FILE = "user_data.json"
MODEL_DIR = "model_artifacts"

# -------------------- Helper Functions --------------------
def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_model_artifacts():
    artifacts = {}
    try:
        artifacts['encoders'] = joblib.load(f'{MODEL_DIR}/encoders.pkl')
        artifacts['feature_names'] = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
        artifacts['target_names'] = joblib.load(f'{MODEL_DIR}/target_names.pkl')
        artifacts['X_train'] = np.load(f'{MODEL_DIR}/X_train.npy', allow_pickle=True)
        return artifacts
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None

def categorize_bmi(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

def get_recommendation_from_model(weight, height, age, gender):
    artifacts = load_model_artifacts()
    if not artifacts:
        return None
    try:
        bmi = weight / ((height / 100) ** 2)
        gender_encoded = artifacts['encoders']['Gender'].transform([gender])[0]
        features = np.array([[weight, height, age, gender_encoded]])
        distances = np.sum(np.abs(artifacts['X_train'] - features), axis=1)
        nearest_idx = np.argmin(distances)
        return artifacts['target_names'][nearest_idx % len(artifacts['target_names'])]
    except:
        return None

# -------------------- Gemini AI Configuration --------------------
def configure_gemini():
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

model = configure_gemini()

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = load_json(USERS_FILE)
        email = request.form.get('email', '').strip()
        if not email or not request.form.get('password'):
            return "Email and password required!"
        if email in users:
            return "Email already registered!"
        users[email] = {
            "name": request.form.get('name', ''),
            "password": request.form['password'],
            "age": request.form.get('age', ''),
            "mobile": request.form.get('mobile', ''),
            "gender": request.form.get('gender', 'other')
        }
        save_json(USERS_FILE, users)
        return redirect(url_for('show_login'))
    return render_template('register.html')

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # (Optional) You could log or email this message:
        print(f"📩 Message received from {name} ({email}): {message}")

        flash("Your message has been sent successfully! 💚")
        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def show_login():
    if request.method == 'POST':
        users = load_json(USERS_FILE)
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if email in users and users[email]['password'] == password:
            session['username'] = email
            return redirect(url_for('dashboard'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('show_login'))
    users = load_json(USERS_FILE)
    name = users.get(session['username'], {}).get('name', 'User')
    return render_template('dashboard.html', name=name)


@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('show_login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/bmi_tracker', methods=['GET', 'POST'])
def bmi_tracker():
    if 'username' not in session:
        return redirect(url_for('show_login'))
    if request.method == 'POST':
        try:
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            age = int(request.form.get('age', 25))
            users = load_json(USERS_FILE)
            gender = users.get(session['username'], {}).get('gender', 'other')
            bmi = round(weight / ((height / 100) ** 2), 1)
            category = categorize_bmi(bmi)
            rec_code = get_recommendation_from_model(weight, height, age, gender)
            return jsonify({"bmi": bmi, "category": category, "recommendation": rec_code})
        except Exception as e:
            return jsonify({"error": str(e)})
    return render_template('bmi_tracker.html')

@app.route('/ai_recommendation', methods=['POST'])
def ai_recommendation():
    try:
        data = request.get_json()
        if not model:
            return jsonify({"error": "AI service unavailable"}), 503
        prompt = f"""Create a personalized fitness plan for:
        - Gender: {data.get('gender')}
        - Age: {data.get('age')}
        - Height: {data.get('height')}cm
        - Weight: {data.get('weight')}kg
        - Goal: {data.get('goal', 'general fitness')} 
        Do not include consultation advice in response."""
        response = model.generate_content(prompt)
        return jsonify({"recommendation": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Diet Planner --------------------
planner = DietPlanner()
planner.load_data(data_dir='data')

@app.route('/diet_planner')
def diet_planner():
    return render_template('diet_planner.html')

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    try:
        data = request.get_json()
        if not data or 'calories' not in data or 'diet_type' not in data:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        calories = int(data['calories'])
        if calories < 1200 or calories > 5000:
            return jsonify({'success': False, 'error': 'Calories must be between 1200 and 5000'}), 400

        options_per_meal = int(data.get('options_per_meal', 2))
        allergens = [a.strip() for a in data.get('allergens', []) if a.strip()]
        
        meal_counts = {
            'Breakfast': 0.5,
            'Morning Snack': 0.1,
            'Lunch': 0.3,
            'Evening Snack': 0.1,
            'Dinner': 0.1
        }

        # ✅ generate plan first
        weekly_plan = planner.generate_weekly_plan(
        daily_calories=calories,
        diet_type=data['diet_type'],
        allergens=allergens,
        options_per_meal=options_per_meal,
    )


        # ✅ then make it serializable
        def make_serializable(plan):
            for day in plan:
                for meal in day['meals']:
                    meal['target_calories'] = float(meal['target_calories'])
                    for option in meal['options']:
                        option['calories'] = int(option['calories'])
                        option['protein'] = float(option['protein'])
                        option['carbs'] = float(option['carbs'])
                        option['fat'] = float(option['fat'])
            return plan

        weekly_plan = make_serializable(weekly_plan)
        

        return jsonify({
            'success': True,
            'plan': weekly_plan,
            'meal_structure': meal_counts
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download-plan', methods=['GET'])
def download_plan():
    try:
        plan_str = request.args.get('plan', '[]')
        weekly_plan = json.loads(plan_str)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 28)
        pdf.cell(200, 10, txt="Nutribyte | Weekly Meal Plan", ln=1, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
        pdf.ln(10)

        for day in weekly_plan:
            pdf.set_font("Arial", 'B', 20)
            pdf.cell(200, 10, txt=day['day'], ln=1)
            pdf.ln(5)
            for meal in day['meals']:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt=f"{meal['meal_name']} (~{int(meal['target_calories'])} cal)", ln=1)
                pdf.set_font("Arial", size=14)
                for i, option in enumerate(meal['options'], 1):
                    pdf.cell(200, 10, txt=f"Option {i}: {option['name']}", ln=1)
                    pdf.cell(200, 10, txt=f"Calories: {int(option['calories'])} | Protein: {option['protein']:.1f}g | Carbs: {option['carbs']:.1f}g | Fat: {option['fat']:.1f}g", ln=1)
                    pdf.ln(3)
                pdf.ln(5)
            pdf.ln(10)

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'Nutribyte_plan.pdf')
        pdf.output(temp_path)

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'Nutribyte_plan_{datetime.now().strftime("%Y%m%d")}.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save-plan', methods=['POST'])
def save_plan():
    try:
        plan_data = request.get_json()
        if not plan_data:
            return jsonify({'success': False, 'error': 'No plan data provided'}), 400
        return jsonify({'success': True, 'message': 'Plan saved successfully (demo)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        food_name = data.get('food_name')
        if not food_name:
            return jsonify({'success': False, 'error': 'No food name provided'})
        model = configure_gemini()
        prompt = f"Give me a healthy oil-free recipe using {food_name}. Keep it simple and shortand healthy too add the calories included in each thing used to make it but atleast complete it"
        response = model.generate_content(prompt)
        return jsonify({'success': True, 'recipe': response.text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_data', methods=['POST'])
def save_data():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 403

    data = request.get_json()
    all_data = load_json(USER_DATA_FILE)
    all_data[session['username']] = data
    save_json(USER_DATA_FILE, all_data)
    return jsonify({'success': True})

@app.route('/get_data', methods=['GET'])
def get_data():
    if 'username' not in session:
        return jsonify({})

    all_data = load_json(USER_DATA_FILE)
    return jsonify(all_data.get(session['username'], {}))


# Ensure upload dir exists
UPLOAD_FOLDER = os.path.join("static", "images", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# --- load your datasets once at startup ---
def _read_csv_safe(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().title() for c in df.columns]
    # Ensure expected columns exist
    expected = ["Food", "Class", "Ingredients", "Calories", "Protein", "Carbs", "Fat", "Healthy"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    # coerce numeric
    for col in ["Calories", "Protein", "Carbs", "Fat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # drop rows without names
    df = df.dropna(subset=["Food"])
    return df

INDIAN = _read_csv_safe(os.path.join("data", "Indian_Food.csv"))
SNACKS = _read_csv_safe(os.path.join("data", "Snacks.csv"))
FRUITS = _read_csv_safe(os.path.join("data", "fruits.csv"))
ALL_DATASETS = pd.concat([INDIAN, SNACKS, FRUITS], ignore_index=True)

# Precompute a lowercase name column for fuzzy matches
ALL_DATASETS["Food_lc"] = ALL_DATASETS["Food"].str.lower().str.strip()

# --- helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prompt_for_json():
    """
    Strict JSON prompt so we can parse reliably.
    """
    return (
        "You are a nutrition assistant. Given a photo of a plate of food,also try to predict the name which is actually known to us please "
        "return a STRICT JSON object (no prose) in this schema:\n"
        "{\n"
        '  "items": [\n'
        '    {"name": "string (food/dish name)", "confidence": 0-1, "serving": "e.g., 1 bowl/1 piece", "quantity": number}\n'
        "  ],\n"
        '  "notes": "short helpful notes (optional)"\n'
        "}\n\n"
        "Rules:\n"
        "- Output ONLY valid in tabular format. No markdown, no backticks.\n"
        "- Use Indian dish names when possible.\n"
        "- If multiple foods on plate, list each as separate item.\n"
        "- If unsure about quantity, set quantity=1 and serving empty.\n make sure to create a table based on category"
    )

def call_gemini_json(image_path):
    model = genai.GenerativeModel("gemini-2.0-flash")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = model.generate_content(
        [
            prompt_for_json(),
            {"mime_type": "image/jpeg", "data": img_bytes}
        ],
        safety_settings=None,
    )
    # Try parsing JSON
    txt = (resp.text or "").strip()
    # Sometimes models wrap JSON in code fences
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        data = json.loads(txt)
        if not isinstance(data, dict) or "items" not in data:
            raise ValueError("Invalid schema")
        return data
    except Exception:
        # Fallback: extract first plausible food word
        m = re.search(r"(?i)([A-Za-z][A-Za-z\s\-]{2,})", txt)
        name = m.group(1).strip() if m else "Unknown food"
        return {"items": [{"name": name, "confidence": 0.4, "serving": "", "quantity": 1}], "notes": "fallback parse"}

def match_food_to_dataset(name, cutoff=0.8):
    """
    Improved deterministic food matcher:
    Handles generic Indian food names (like 'dal', 'rice', 'roti') more intelligently.
    """
    if not name:
        return None

    name_lc = name.lower().strip()

    # --- Generic base-name resolver ---
    GENERIC_GROUPS = {
        "dal": ["toor dal", "moong dal", "masoor dal", "urad dal", "chana dal"],
        "rice": ["plain rice", "boiled rice", "steamed rice", "basmati rice", "white rice"],
        "roti": ["roti", "chapati", "phulka"],
        "sabji": ["mixed vegetable", "vegetable curry", "bhindi masala", "aloo gobi"],
        "curry": ["vegetable curry", "paneer curry", "chicken curry"],
    }

    # If a generic name is predicted (e.g., "dal"), restrict dataset search to that family
    matched_group = None
    for key, variants in GENERIC_GROUPS.items():
        if key in name_lc:
            matched_group = variants
            break

    # --- 1. Exact match ---
    exact = ALL_DATASETS[ALL_DATASETS["Food_lc"] == name_lc]
    if not exact.empty:
        exact_sorted = exact.sort_values(by="Food_lc", key=lambda x: x.str.len(), ascending=False)
        return exact_sorted.iloc[0].to_dict()

    # --- 2. Word-boundary or group match ---
    subset = ALL_DATASETS
    if matched_group:
        subset = ALL_DATASETS[ALL_DATASETS["Food_lc"].isin(matched_group)]
    regex = rf"\b{name_lc}\b"
    word_match = subset[subset["Food_lc"].str.contains(regex, na=False, regex=True)]
    if not word_match.empty:
        word_match["overlap"] = word_match["Food_lc"].apply(
            lambda x: len(set(name_lc.split()) & set(x.split()))
        )
        best_row = word_match.sort_values(["overlap", "Food_lc"], ascending=[False, True]).iloc[0]
        return best_row.to_dict()

    # --- 3. Fuzzy match with preference for base foods ---
    from difflib import SequenceMatcher
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    subset["sim_score"] = subset["Food_lc"].apply(lambda x: similarity(name_lc, x))
    best_row = subset[subset["sim_score"] >= cutoff].sort_values(
        by=["sim_score", "Food_lc"], ascending=[False, True]
    )

    if not best_row.empty:
        # Prefer base variants like "plain rice" or "toor dal" over "brown rice"
        if matched_group:
            # Select the most common/base variant first
            for variant in matched_group:
                match = best_row[best_row["Food_lc"].str.contains(variant, case=False, regex=False)]
                if not match.empty:
                    return match.iloc[0].to_dict()
        # Otherwise, default to best similarity
        return best_row.iloc[0].to_dict()

    # --- 4. Not found ---
    return None




def compute_totals(items):
    """
    items: list of dicts with numeric macros; returns totals dict.
    """
    totals = {"Calories": 0.0, "Protein": 0.0, "Carbs": 0.0, "Fat": 0.0}
    for it in items:
        qty = float(it.get("quantity", 1) or 1)
        totals["Calories"] += (it.get("Calories") or 0) * qty
        totals["Protein"]  += (it.get("Protein")  or 0) * qty
        totals["Carbs"]    += (it.get("Carbs")    or 0) * qty
        totals["Fat"]      += (it.get("Fat")      or 0) * qty
    # Round nicely
    for k in totals:
        totals[k] = round(totals[k], 2)
    return totals

# --- routes ---
@app.route("/image-calorie", methods=["GET"])
def image_calorie_page():
    return render_template("image_calorie.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "food_image" not in request.files:
        flash("No file part.")
        return redirect(url_for("image_calorie_page"))

    file = request.files["food_image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("image_calorie_page"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG/JPG/JPEG/WEBP/BMP/GIF.")
        return redirect(url_for("image_calorie_page"))

    # Save upload
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Call Gemini to identify foods
    gemini_out = call_gemini_json(filepath)
    raw_items = gemini_out.get("items", [])
    notes = gemini_out.get("notes", "")

    # Map each detected item to your dataset
    matched_items = []
    for it in raw_items:
        name = (it.get("name") or "").strip()
        qty = it.get("quantity") or 1
        serving = it.get("serving") or ""
        conf = it.get("confidence") or 0

        row = match_food_to_dataset(name)
        if row:
            matched_items.append({
                "DetectedName": name,
                "MatchedFood": row["Food"],
                "Serving": serving,
                "Quantity": qty,
                "Confidence": round(conf, 2),
                "Calories": row["Calories"],
                "Protein": row["Protein"],
                "Carbs": row["Carbs"],
                "Fat": row["Fat"],
                "Class": row["Class"]
            })
        else:
            matched_items.append({
                "DetectedName": name,
                "MatchedFood": "Not found in dataset",
                "Serving": serving,
                "Quantity": qty,
                "Confidence": round(conf, 2),
                "Calories": None,
                "Protein": None,
                "Carbs": None,
                "Fat": None,
                "Class": None
            })

            

    totals = compute_totals([m for m in matched_items if m["Calories"] is not None])

    return render_template(
        "image_result.html",
        image_url="/" + filepath.replace("\\", "/"),
        items=matched_items,
        totals=totals,
        notes=notes
    )

# --- imports ---




# -------------------- OCR CONFIG --------------------
HARMFUL_KEYWORDS = [
    "msg", "monosodium glutamate", "trans fat", "hydrogenated",
    "aspartame", "acesulfame k", "high fructose", "maida", "refined flour"
]

# On Windows, specify Tesseract path if needed
if os.name == "nt":
    # You can adjust this if your Tesseract is installed elsewhere
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# -------------------- OCR Routes --------------------
@app.route("/ocr")
def ocr_page():
    """Render the OCR upload page."""
    if "username" not in session:
        return redirect(url_for("show_login"))
    return render_template("ocr.html")


@app.route("/extract_text", methods=["POST"])
def extract_text():
    """Handles OCR extraction, harmful detection, and Gemini analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # --- Run OCR safely ---
        with Image.open(filepath) as img:
            text = pytesseract.image_to_string(img)

        if not text.strip():
            return jsonify({"error": "No readable text found in image."}), 400

        # --- Detect harmful ingredients ---
        harmful_found = [
            word for word in HARMFUL_KEYWORDS if word.lower() in text.lower()
        ]

        # --- Gemini analysis ---
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        You are a food safety assistant.
        These are the extracted ingredients/nutritional details from a food packet:

        {text}

        1. Identify harmful or unhealthy ingredients (if any).
        2. Give a short summary: Is this product safe to eat regularly or only occasionally?
        3. Mention why in 2-3 simple sentencesbut in points each in a seperate line.
        """

        gemini_response = model.generate_content(prompt)
        gemini_analysis = getattr(gemini_response, "text", "").strip() or "No response from Gemini."

        result = {
            "extracted_text": text.strip(),
            "harmful_keywords": harmful_found if harmful_found else ["No harmful ingredients detected"],
            "gemini_analysis": gemini_analysis
        }

        return jsonify(result)

    except Exception as e:
        # Catch all unexpected issues
        return jsonify({"error": f"OCR processing failed: {str(e)}"}), 500

    finally:
        # Safe cleanup
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass



# -------------------- Run App --------------------
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True, port=5000)

