import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'decision.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML Template with inline CSS styling (ambient dark theme)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Prediction</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #07060f;
            --bg-mid: #0d0b1e;
            --glass: rgba(20, 18, 40, 0.55);
            --glass-border: rgba(255, 255, 255, 0.08);
            --field-bg: rgba(255, 255, 255, 0.04);
            --field-border: rgba(255, 255, 255, 0.10);
            --text: #e6e4f5;
            --text-muted: #8c89ab;
            --accent: #8b7cff;
            --accent-2: #4cc9f0;
            --accent-glow: rgba(139, 124, 255, 0.45);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: radial-gradient(ellipse at 20% 0%, var(--bg-mid) 0%, var(--bg-deep) 60%);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 24px;
            overflow-x: hidden;
            position: relative;
        }

        /* ---------- Ambient floating glow orbs ---------- */
        .ambient {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.55;
            animation: drift 22s ease-in-out infinite alternate;
        }

        .orb.one {
            width: 460px; height: 460px;
            background: #5b3df5;
            top: -140px; left: -120px;
        }

        .orb.two {
            width: 520px; height: 520px;
            background: #0ea5c9;
            bottom: -200px; right: -140px;
            animation-duration: 28s;
            animation-delay: -6s;
            opacity: 0.38;
        }

        .orb.three {
            width: 340px; height: 340px;
            background: #c026d3;
            top: 45%; left: 55%;
            animation-duration: 34s;
            animation-delay: -12s;
            opacity: 0.22;
        }

        @keyframes drift {
            0%   { transform: translate(0, 0) scale(1); }
            50%  { transform: translate(60px, 40px) scale(1.12); }
            100% { transform: translate(-40px, 80px) scale(0.95); }
        }

        /* subtle film-grain / star dust */
        .ambient::after {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                radial-gradient(1px 1px at 12% 22%, rgba(255,255,255,0.35), transparent),
                radial-gradient(1px 1px at 78% 14%, rgba(255,255,255,0.25), transparent),
                radial-gradient(1.5px 1.5px at 34% 78%, rgba(255,255,255,0.3), transparent),
                radial-gradient(1px 1px at 90% 66%, rgba(255,255,255,0.25), transparent),
                radial-gradient(1px 1px at 55% 40%, rgba(255,255,255,0.2), transparent),
                radial-gradient(1.5px 1.5px at 8% 60%, rgba(255,255,255,0.25), transparent);
            animation: twinkle 6s ease-in-out infinite alternate;
        }

        @keyframes twinkle {
            from { opacity: 0.35; }
            to   { opacity: 0.9; }
        }

        /* ---------- Glass card ---------- */
        .container {
            position: relative;
            z-index: 1;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            backdrop-filter: blur(22px) saturate(140%);
            -webkit-backdrop-filter: blur(22px) saturate(140%);
            box-shadow:
                0 0 0 1px rgba(255, 255, 255, 0.02) inset,
                0 30px 80px rgba(0, 0, 0, 0.55),
                0 0 80px rgba(139, 124, 255, 0.08);
            max-width: 820px;
            width: 100%;
            padding: 40px 36px 36px;
            animation: rise 0.9s cubic-bezier(0.2, 0.7, 0.2, 1) both;
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(24px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
            background: linear-gradient(120deg, #c4baff 0%, var(--accent) 40%, var(--accent-2) 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 32px;
            font-size: 0.92rem;
            font-weight: 300;
            letter-spacing: 0.02em;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 500;
            margin-bottom: 7px;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: var(--text-muted);
        }

        input, select {
            color-scheme: dark;
            padding: 11px 14px;
            background: var(--field-bg);
            color: var(--text);
            border: 1px solid var(--field-border);
            border-radius: 10px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
        }

        input::placeholder {
            color: rgba(140, 137, 171, 0.5);
        }

        select {
            appearance: none;
            -webkit-appearance: none;
            cursor: pointer;
            padding-right: 38px;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1.5l5 5 5-5' fill='none' stroke='%238c89ab' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>");
            background-repeat: no-repeat;
            background-position: right 14px center;
        }

        select option {
            background: #14122a;
            color: var(--text);
        }

        input:hover, select:hover {
            border-color: rgba(255, 255, 255, 0.2);
            background-color: rgba(255, 255, 255, 0.06);
        }

        input:focus, select:focus {
            border-color: var(--accent);
            background-color: rgba(139, 124, 255, 0.07);
            box-shadow: 0 0 0 3px rgba(139, 124, 255, 0.18), 0 0 22px rgba(139, 124, 255, 0.2);
        }

        /* staggered fade-in for fields */
        .form-group {
            animation: fadeIn 0.7s ease both;
        }
        .form-group:nth-child(1)  { animation-delay: 0.10s; }
        .form-group:nth-child(2)  { animation-delay: 0.14s; }
        .form-group:nth-child(3)  { animation-delay: 0.18s; }
        .form-group:nth-child(4)  { animation-delay: 0.22s; }
        .form-group:nth-child(5)  { animation-delay: 0.26s; }
        .form-group:nth-child(6)  { animation-delay: 0.30s; }
        .form-group:nth-child(7)  { animation-delay: 0.34s; }
        .form-group:nth-child(8)  { animation-delay: 0.38s; }
        .form-group:nth-child(9)  { animation-delay: 0.42s; }
        .form-group:nth-child(10) { animation-delay: 0.46s; }
        .form-group:nth-child(11) { animation-delay: 0.50s; }
        .form-group:nth-child(12) { animation-delay: 0.54s; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Button ---------- */
        .submit-btn {
            grid-column: 1 / -1;
            position: relative;
            background: linear-gradient(120deg, #6d5bf7, #8b7cff 50%, #4cc9f0);
            background-size: 200% 100%;
            color: #fff;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            cursor: pointer;
            margin-top: 10px;
            box-shadow: 0 8px 30px var(--accent-glow);
            transition: transform 0.2s ease, box-shadow 0.3s ease, background-position 0.5s ease;
        }

        .submit-btn:hover {
            background-position: 100% 0;
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(139, 124, 255, 0.6);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* ---------- Result ---------- */
        .result-box {
            margin-top: 28px;
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            animation: rise 0.7s cubic-bezier(0.2, 0.7, 0.2, 1) both;
        }

        .result-placed {
            background: rgba(16, 185, 129, 0.10);
            color: #6ee7b7;
            border: 1px solid rgba(52, 211, 153, 0.35);
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.18), inset 0 0 30px rgba(16, 185, 129, 0.06);
        }

        .result-not-placed {
            background: rgba(244, 63, 94, 0.10);
            color: #fda4af;
            border: 1px solid rgba(251, 113, 133, 0.35);
            box-shadow: 0 0 40px rgba(244, 63, 94, 0.18), inset 0 0 30px rgba(244, 63, 94, 0.06);
        }

        .result-error {
            background: rgba(245, 158, 11, 0.10);
            color: #fcd34d;
            border: 1px solid rgba(251, 191, 36, 0.35);
            font-size: 0.95rem;
            font-weight: 500;
        }

        @media (max-width: 520px) {
            .container { padding: 30px 20px 26px; }
            h1 { font-size: 1.6rem; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation: none !important; transition: none !important; }
        }
    </style>
</head>
<body>

<div class="ambient">
    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>
</div>

<div class="container">
    <h1>Placement Predictor</h1>
    <p class="subtitle">Enter candidate details to predict campus placement status</p>

    <form action="/predict" method="POST">
        <div class="grid-form">
            <!-- Gender -->
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="M">Male (M)</option>
                    <option value="F">Female (F)</option>
                </select>
            </div>

            <!-- SSC Percentage -->
            <div class="form-group">
                <label for="ssc_p">10th Percentage (ssc_p)</label>
                <input type="number" step="0.01" id="ssc_p" name="ssc_p" placeholder="e.g. 67.00" required>
            </div>

            <!-- SSC Board -->
            <div class="form-group">
                <label for="ssc_b">10th Board (ssc_b)</label>
                <select id="ssc_b" name="ssc_b" required>
                    <option value="Central">Central</option>
                    <option value="Others">Others</option>
                </select>
            </div>

            <!-- HSC Percentage -->
            <div class="form-group">
                <label for="hsc_p">12th Percentage (hsc_p)</label>
                <input type="number" step="0.01" id="hsc_p" name="hsc_p" placeholder="e.g. 70.00" required>
            </div>

            <!-- HSC Board -->
            <div class="form-group">
                <label for="hsc_b">12th Board (hsc_b)</label>
                <select id="hsc_b" name="hsc_b" required>
                    <option value="Central">Central</option>
                    <option value="Others">Others</option>
                </select>
            </div>

            <!-- HSC Stream -->
            <div class="form-group">
                <label for="hsc_s">12th Stream (hsc_s)</label>
                <select id="hsc_s" name="hsc_s" required>
                    <option value="Commerce">Commerce</option>
                    <option value="Science">Science</option>
                    <option value="Arts">Arts</option>
                </select>
            </div>

            <!-- Degree Percentage -->
            <div class="form-group">
                <label for="degree_p">Degree Percentage (degree_p)</label>
                <input type="number" step="0.01" id="degree_p" name="degree_p" placeholder="e.g. 58.00" required>
            </div>

            <!-- Degree Type -->
            <div class="form-group">
                <label for="degree_t">Degree Type (degree_t)</label>
                <select id="degree_t" name="degree_t" required>
                    <option value="Sci&amp;Tech">Sci&amp;Tech</option>
                    <option value="Comm&amp;Mgmt">Comm&amp;Mgmt</option>
                    <option value="Others">Others</option>
                </select>
            </div>

            <!-- Work Experience -->
            <div class="form-group">
                <label for="workex">Work Experience (workex)</label>
                <select id="workex" name="workex" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <!-- Employability Test Percentage -->
            <div class="form-group">
                <label for="etest_p">E-Test Percentage (etest_p)</label>
                <input type="number" step="0.01" id="etest_p" name="etest_p" placeholder="e.g. 55.00" required>
            </div>

            <!-- Specialisation -->
            <div class="form-group">
                <label for="specialisation">MBA Specialization</label>
                <select id="specialisation" name="specialisation" required>
                    <option value="Mkt&amp;HR">Mkt&amp;HR</option>
                    <option value="Mkt&amp;Fin">Mkt&amp;Fin</option>
                </select>
            </div>

            <!-- MBA Percentage -->
            <div class="form-group">
                <label for="mba_p">MBA Percentage (mba_p)</label>
                <input type="number" step="0.01" id="mba_p" name="mba_p" placeholder="e.g. 58.80" required>
            </div>

            <button type="submit" class="submit-btn">Predict Placement Status</button>
        </div>
    </form>

    {% if prediction %}
        <div class="result-box {% if prediction == 'Placed' %}result-placed{% elif prediction == 'Not Placed' %}result-not-placed{% else %}result-error{% endif %}">
            Prediction Result: {{ prediction }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model file 'decision.pkl' not found.", 500

    try:
        # Standard Label Encoding maps typically used for the Campus Recruitment dataset:
        # gender: M -> 1, F -> 0
        gender_map = {'M': 1, 'F': 0}
        # boards / workex / specialisation: Central/Yes/Mkt&Fin -> 1, Others/No/Mkt&HR -> 0, etc.
        binary_map = {'Central': 0, 'Others': 1, 'Yes': 1, 'No': 0, 'Mkt&Fin': 1, 'Mkt&HR': 0}

        # hsc_s: Commerce -> 1, Science -> 2, Arts -> 0 (or dependent on how LabelEncoder alphabetized them)
        # Alphabetical sorting for LabelEncoder: Arts(0), Commerce(1), Science(2)
        hsc_s_map = {'Arts': 0, 'Commerce': 1, 'Science': 2}

        # degree_t: Comm&Mgmt -> 0, Others -> 1, Sci&Tech -> 2 (Alphabetical sorting)
        degree_t_map = {'Comm&Mgmt': 0, 'Others': 1, 'Sci&Tech': 2}

        # Retrieve and encode form data
        input_data = {
            'gender': gender_map.get(request.form.get('gender'), 1),
            'ssc_p': float(request.form.get('ssc_p', 0)),
            'ssc_b': binary_map.get(request.form.get('ssc_b'), 0),
            'hsc_p': float(request.form.get('hsc_p', 0)),
            'hsc_b': binary_map.get(request.form.get('hsc_b'), 0),
            'hsc_s': hsc_s_map.get(request.form.get('hsc_s'), 1),
            'degree_p': float(request.form.get('degree_p', 0)),
            'degree_t': degree_t_map.get(request.form.get('degree_t'), 0),
            'workex': binary_map.get(request.form.get('workex'), 0),
            'etest_p': float(request.form.get('etest_p', 0)),
            'specialisation': binary_map.get(request.form.get('specialisation'), 0),
            'mba_p': float(request.form.get('mba_p', 0))
        }

        # Convert input data to DataFrame matching original feature order
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        # If the model returns an encoded number (1 = Placed, 0 = Not Placed),
        # turn it into a readable label so the result box styles correctly.
        if not isinstance(prediction, str):
            prediction = 'Placed' if int(prediction) == 1 else 'Not Placed'
    except Exception as e:
        prediction = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)