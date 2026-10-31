import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Decision Tree model safely using absolute path
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'decision.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# HTML & CSS Template combined into a single responsive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Predictor</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-color: #0f172a;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 700px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 25px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            margin-bottom: 12px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .Approved { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
        .Rejected { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
            .form-group.full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Loan Approval Status Predictor</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="no_of_dependents">Number of Dependents</label>
                    <input type="number" id="no_of_dependents" name="no_of_dependents" required min="0" placeholder="e.g. 2">
                </div>
                
                <!-- Categorical Field: Education -->
                <div class="form-group">
                    <label for="education">Education</label>
                    <select id="education" name="education" required>
                        <option value="0">Graduate</option>
                        <option value="1">Not Graduate</option>
                    </select>
                </div>

                <!-- Categorical Field: Self Employed -->
                <div class="form-group">
                    <label for="self_employed">Self Employed</label>
                    <select id="self_employed" name="self_employed" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="income_annum">Annual Income (₹)</label>
                    <input type="number" step="any" id="income_annum" name="income_annum" required min="0" placeholder="e.g. 5000000">
                </div>
                <div class="form-group">
                    <label for="loan_amount">Loan Amount (₹)</label>
                    <input type="number" step="any" id="loan_amount" name="loan_amount" required min="0" placeholder="e.g. 15000000">
                </div>
                <div class="form-group">
                    <label for="cibil_score">CIBIL Score</label>
                    <input type="number" id="cibil_score" name="cibil_score" required min="300" max="900" placeholder="e.g. 750">
                </div>
                <div class="form-group">
                    <label for="residential_assets_value">Residential Assets Value (₹)</label>
                    <input type="number" step="any" id="residential_assets_value" name="residential_assets_value" required min="0" placeholder="e.g. 2000000">
                </div>
                <div class="form-group">
                    <label for="commercial_assets_value">Commercial Assets Value (₹)</label>
                    <input type="number" step="any" id="commercial_assets_value" name="commercial_assets_value" required min="0" placeholder="e.g. 0">
                </div>
                <div class="form-group">
                    <label for="luxury_assets_value">Luxury Assets Value (₹)</label>
                    <input type="number" step="any" id="luxury_assets_value" name="luxury_assets_value" required min="0" placeholder="e.g. 500000">
                </div>
                <div class="form-group">
                    <label for="bank_asset_value">Bank Asset Value (₹)</label>
                    <input type="number" step="any" id="bank_asset_value" name="bank_asset_value" required min="0" placeholder="e.g. 1000000">
                </div>
            </div>
            
            <button type="submit">Check Loan Status</button>
        </form>

        {% if prediction %}
        <div class="result-card {{ prediction }}">
            Loan Status: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the exact order expected by the decision tree
            features = [
                float(request.form['no_of_dependents']),
                float(request.form['education']),
                float(request.form['self_employed']),
                float(request.form['income_annum']),
                float(request.form['loan_amount']),
                float(request.form['cibil_score']),
                float(request.form['residential_assets_value']),
                float(request.form['commercial_assets_value']),
                float(request.form['luxury_assets_value']),
                float(request.form['bank_asset_value'])
            ]
            
            input_data = np.array([features])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
