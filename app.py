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

# HTML Template with inline CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Prediction</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 100%;
            padding: 30px;
        }

        h1 {
            text-align: center;
            margin-bottom: 8px;
            color: var(--primary-color);
        }

        p.subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 25px;
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 0.88rem;
            color: #374151;
        }

        input, select {
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
        }

        .submit-btn {
            grid-column: 1 / -1;
            background-color: var(--primary-color);
            color: white;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease;
            margin-top: 10px;
        }

        .submit-btn:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }

        .result-placed {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }

        .result-not-placed {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
    </style>
</head>
<body>

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
                    <option value="Sci&Tech">Sci&Tech</option>
                    <option value="Comm&Mgmt">Comm&Mgmt</option>
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
                    <option value="Mkt&HR">Mkt&HR</option>
                    <option value="Mkt&Fin">Mkt&Fin</option>
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
        <div class="result-box {% if prediction == 'Placed' %}result-placed{% else %}result-not-placed{% endif %}">
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
    except Exception as e:
        prediction = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)